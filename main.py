import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import sys
from PIL import Image
import pytesseract
import os

sys.path.append('E:\Text_model\src')

from ocr.helpers import implt, resize
from ocr import page
from ocr import words


IMG = '1'    # 1, 2, 3
filename ='E:/Medical-Prescription-OCR-master/Model-2/test/3.png'
#save_filename = "E:/Medical-Prescription-OCR-master/Model-2/test/2_1.jpg"

image=cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
implt(image)
#crop = page.detection(image)
#implt(crop)
#cv2.imwrite('E:\Text_model\images\hhha.jpg',crop)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
gray = cv2.medianBlur(gray, 3)
implt(gray)

cv2.imwrite("E:\Text_model\images\dfsdf.jpg", gray)

text = pytesseract.image_to_string(gray)
#os.remove(save_filename)
print(text)
