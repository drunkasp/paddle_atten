import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class CBAMLayer(nn.Layer):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩 H,W 为 1
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2D(channel, channel // reduction, 1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(channel // reduction, channel, 1, bias_attr=False)
        )

        # spatial attention
        self.conv = nn.Conv2D(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        # Spatial Attention
        max_out = paddle.max(x, axis=1, keepdim=True)
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(paddle.concat([max_out, avg_out], axis=1)))
        x = spatial_out * x

        return x

# 测试
x = paddle.randn([1, 1024, 32, 32])
net = CBAMLayer(1024)
y = net(x)
print(y.shape)
