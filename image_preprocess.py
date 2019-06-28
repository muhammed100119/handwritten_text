import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
sys.path.append('../src')
from ocr.helpers import implt, resize, ratio
import pytesseract
#%matplotlib inline
plt.rcParams['figure.figsize'] = (9.0, 9.0)
image = cv2.cvtColor(cv2.imread('E:/NLP-master/images/claim_Byju member vst_0.jpg'), cv2.COLOR_BGR2RGB)
implt(image)

def edges_det(img, min_val, max_val):
    """ Preprocessing (gray, thresh, filter, border) + Canny edge detection """
    kernel = np.ones((2, 2), np.uint8)
    
    img = cv2.cvtColor(resize(img), cv2.COLOR_BGR2GRAY)

#    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.GaussianBlur(img, (5, 5), 0)
#    img = cv2.GaussianBlur(img, (5, 5), 0)
    implt(img,'gray','blur')
    img = cv2.adaptiveThreshold(img,175, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    implt(img, 'gray', 'Adaptive Threshold')
    img = cv2.erode(img, kernel, iterations=1)
    implt(img,'gray','erode')
    img = cv2.dilate(img, kernel, iterations=1)
    implt(img,'gray','dilate')
    img = cv2.medianBlur(img, 1)
#    img = cv2.blur(img,(10,10))
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    implt(img, 'gray', 'Median Blur + Border')
    return cv2.Canny(img, min_val, max_val)

edges_image = edges_det(image, 200, 250)
edges_image = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, np.ones((5, 11)))
implt(edges_image, 'gray', 'Edges')

def write_file(text,image):
    cv2.imwrite("E:/Text_model/claim_page_image_preprocessed.jpg",image)
    text_file = open('E:/Text_model/claim_page_Text.txt',"w")  
    text_file.write(text)
    text_file.close()
    
def four_corners_sort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right"""
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def contour_offset(cnt, offset):
    """ Offset contour because of 5px border """
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt


def find_page_contours(edges, img):
    """ Finding corner points of page contour """
    # Getting contours  
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Finding biggest rectangle otherwise return original corners
    height = edges.shape[0]
    width = edges.shape[1]
    MIN_COUNTOUR_AREA = height * width * 0.5
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

    max_area = MIN_COUNTOUR_AREA
    page_contour = np.array([[0, 0],
                            [0, height-5],
                            [width-5, height-5],
                            [width-5, 0]])

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        # Page has 4 corners and it is convex
        if (len(approx) == 4 and
                cv2.isContourConvex(approx) and
                max_area < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):
            
            max_area = cv2.contourArea(approx)
            page_contour = approx[:, 0]

    # Sort corners and offset them
    page_contour = four_corners_sort(page_contour)
    return contour_offset(page_contour, (-5, -5))

page_contour = find_page_contours(edges_image, resize(image))
print("PAGE CONTOUR:")
print(page_contour)
implt(cv2.drawContours(resize(image), [page_contour], -1, (0, 255, 0), 3))

      
#Write To File
text=pytesseract.image_to_string(image)
print(text)
write_file(text,image)

