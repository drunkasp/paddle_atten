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
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 只让程序看到 GPU3

image_path = '/workspace/project/dataset/xhs/subsample2'

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

result = ocr.predict(image_path)
for res in result:
    res.save_to_img("pre_output/subsample2")

# 4+7+5=16/125