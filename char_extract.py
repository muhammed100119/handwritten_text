import os
import cv2
import pytesseract
from pytesseract import Output
import skimage.io
import matplotlib.pyplot as plt

count=1001
image='E:/Text_model/images/imag.png'
basename=os.path.basename(image.split('.')[0])
# read the image and get the dimensions
img = cv2.imread(image)
h, w, _ = img.shape # assumes color image

# run tesseract, returning the bounding boxes
boxes = pytesseract.image_to_boxes(img) # also include any config options you use


# draw the bounding boxes on the image
for b in boxes.splitlines():
    b = b.split(' ')
    char_image = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
#    x1=int(b[1])
#    y1=h - int(b[2])
#    w1=int(b[3])
#    h1=h - int(b[4])
#    char_crop=char_image[y1:y1+h1, x1:x1+w1]    
    skimage.io.imsave('E:/Text_model/crop/char_crop__' + basename + '_' + str(count) + '.jpg',char_image)
#    plt.imshow(char_image)
#    plt.show()
    count=count+1
#box=[]   
#extracted_list=[]
d = pytesseract.image_to_data(char_image, output_type=Output.DICT)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    crop= char_image[y:y+h, x:x+w]
    plt.imshow(crop)
    plt.show()
#    text=pytesseract.image_to_string(crop)
#    print(text, (x, y, w, h))
#    extracted_list.append(text)
#    box.append((x,y,w,h))
#    draw_rectangle(board,x, y, w, h,"green")
#    turtle.done()
    skimage.io.imsave('E:/Text_model/char_cropped/char_crop___' + basename + '_' + str(count) + '.jpg',crop)
    plt.imshow(crop)
    plt.show()
    count=count+1
    
    
# show annotated image and wait for keypress
cv2.imshow(image, img)
cv2.waitKey(0)