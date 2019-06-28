import os
import shutil
import cv2 
import numpy as np 
import pytesseract
#import image 
dirpath='E:/Text_model/rois'

if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)
     
image='E:/Text_model/images/claim_0.jpg'
basename=os.path.basename(image.split('.')[0])
image = cv2.imread(image)
image=cv2.resize(image,(450,500),fx=2.5,fy=2.5)
#grayscale 
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 

#binary 
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV) 

#dilation 
kernel = np.ones((1,1), np.uint8) 
img_dilation = cv2.dilate(thresh, kernel, iterations=1) 

#Find Contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
 
for i, ctr in enumerate(sorted_ctrs): 
    # Get bounding box 
    x, y, w, h = cv2.boundingRect(ctr) 

    # Getting ROI 
    roi = image[y:y+h, x:x+w] 
    # show ROI 
#    cv2.imshow('segment no:'+str(i),roi) 
    cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2) 

    if not os.path.exists('E:/Text_model/rois'):
            os.makedirs('E:/Text_model/rois')
    if w > 15 and h > 15: 
            cv2.imwrite('E:\\Text_model\\rois\\{}.png'.format(i), roi)

cv2.imshow('marked areas',image) 
cv2.imwrite("E:/Text_model/roifull_"+basename+".jpg",image)
cv2.waitKey(0)