import cv2
import numpy as np
# img=cv2.imread('308.jpg',cv2.IMREAD_GRAYSCALE)

def formatImg(img):
    img=cv2.resize(img,(28,28))
    img=255-img[:,:]
    img=(img[:,:]/255).astype('float32')
    # print(img)
    return img
    # cv2.imshow('resize',img)
    # cv2.waitKey()