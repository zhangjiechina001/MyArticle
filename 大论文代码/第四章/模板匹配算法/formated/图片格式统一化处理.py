import cv2
import numpy as np

def formatImg(file_name):
    img=cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    h,w=img.shape
    mask=np.ones((100,100),dtype='u1')
    mask[:,:]=255
    mask[(100-h)//2:(100-h)//2+h,(100-w)//2:(100-w)//2+w]=img
    return mask


import os
files=os.listdir('cutedImg\\')
for file in files:
    if str(file).__contains__('jpg'):
        format=formatImg(file_name='cutedImg\\'+str(file))
        cv2.imshow(file,format)
        cv2.imwrite('cut_format//'+str(file),format)

cv2.waitKey()