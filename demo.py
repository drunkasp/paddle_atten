from paddleocr import PaddleOCR
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re

#降低det的一些率
#对rec坐果率——角度，高宽比例
#对一些识别结果的组合，进行过滤
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只让程序看到 GPU3

image_path = '/workspace/project/dataset/xhs/无效图片/'
def is_only_digits(text):
    return bool(re.match(r'^[0-9]+$', text))

def is_only_letters(text):
    return bool(re.match(r'^[a-zA-Z]+$', text))

def is_single_chinese(text):
    if len(text) == 1:
        if  '\u4e00' <= text <= '\u9fff':
            return True
    return False

def calculate_angle(box):
    x1, y1 = box[0]
    x2, y2 = box[1]
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)  # 弧度制
    angle_deg = math.degrees(angle_rad)  # 转为角度制
    return angle_deg

def single_no_word(texts):
    # 判断是否只有一个字符，且不是字母或数字
    if len(texts) == 1 and not re.match(r'[a-zA-Z0-9]', texts):
        return True
    return False

def calculate_len_heigh(box):
    x1, y1 = box[0]
    x2, y2 = box[2]
    dx = x2 - x1
    dy = y2 - y1
    return  dy/dx   

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_textline_orientation=False,  # 不做角度分类
    lang='ch',
    use_doc_unwarping=False,
    ocr_version = 'PP-OCRv5',
    text_det_limit_side_len=1280,
    text_detection_model_dir='/workspace/project/PaddleOCR/models/PP-OCRv5_server_det_infer/',
    text_det_thresh=0.5,
    #text_det_box_thresh=0.3,
    text_recognition_model_dir='/workspace/project/PaddleOCR/models/PP-OCRv5_server_rec_infer/',
    text_rec_score_thresh=0.73, #0.73
)

def load_text(blank_img, rec_texts, rec_scores, 
              margin_top=10, line_height=70, 
              font_path="/workspace/project/PaddleOCR/char/wqy-microhei.ttc", 
              font_size=60, font_color=(0, 0, 0)):
    # 将 numpy 数组转为 PIL Image
    pil_img = Image.fromarray(blank_img)

    draw = ImageDraw.Draw(pil_img)

    # 加载字体
    font = ImageFont.truetype(font_path, font_size)

    for i, (rec_text, score) in enumerate(zip(rec_texts, rec_scores)):
        text = f"{rec_text}:{score:.2f}"  # 格式化分数为两位小数
        x, y = 10, margin_top + i * line_height
        draw.text((x, y), text, font=font, fill=font_color)

    # 转回 numpy 数组格式
    blank_img = np.array(pil_img)
    return blank_img

image_path = '/workspace/project/dataset/xhs/无效图片/'
out_path = '/workspace/project/PaddleOCR/output/无效图片0.3'
result = ocr.predict(image_path,use_doc_unwarping=False)#,use_doc_preprocessor=False

for res in result:
    input_path = res['input_path']
    name = input_path.split('/')[-1]
    dt_polys = res['dt_polys']#经过置信度过滤后的结果
    rec_texts = res['rec_texts']
    rec_scores = res['rec_scores']
    rec_polys = res['rec_polys']

    # 画框
    img = cv2.imread(input_path)

    #过滤
    filter_points = []
    filter_texts = []
    for points,text in zip(rec_polys,rec_texts):
        if len(text)==1 and not is_single_chinese(text):
            continue
        if calculate_len_heigh(points) > 1.5 or abs(calculate_angle(points)) > 7:
            continue
        filter_points.append(points)
        filter_texts.append(text)

    #画框
    for points in filter_points:
        points = points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img, [points], isClosed=True, color=(255,0,0), thickness=2)
    h, w = img.shape[:2]
    blank_img = 255 * np.ones((h, w, 3), dtype=np.uint8)

    # 在空白图上写文字
    blank_img = load_text(blank_img,filter_texts, rec_scores)
    
    # 拼接图片
    img_expanded = np.concatenate((img, blank_img), axis=1)
    cv2.imwrite(os.path.join(out_path,name), img_expanded)
    print('success')


    # rec_boxes = res['rec_boxes']
    # det_scores = res['det_scores'] 没有这个字段
    # res.save_to_img("/workspace/project/PaddleOCR/output/try")
    # res.save_to_json("/workspace/project/PaddleOCR/output/try")
    # print(dt_polys,rec_texts,rec_scores,rec_boxes)

# +1+3