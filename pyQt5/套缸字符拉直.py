# -*- coding: utf-8 -*-

import cv2
import numpy as np
def kernelChange(x):
    global binary
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,ksize=(x,x))
    result=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
    return result
    tempzeros=np.zeros(result.shape,dtype=np.uint8)
    # maxContoue=FindBigestContour(result)
    # result=cv2.drawContours(result,maxContoue,-1,(255),3)
    # cv2.imshow('binary',result)

def FindBigestContour(src):
    imax = 0
    imaxcontours = -1
    # 返回的是原图片，边界集合，轮廓的属性
    image, contours, hierarchy = cv2.findContours(src,
                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        itemp = cv2.contourArea(contours[i])
        if (imaxcontours < itemp):
            imaxcontours = itemp
            imax = i

    return contours[imax]

img = cv2.imread('09_30_13.jpg')
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
zeros=np.zeros(shape=img.shape,dtype=np.uint8)
ret,binary=cv2.threshold(img,90,255,cv2.THRESH_BINARY)
cv2.namedWindow('binary',cv2.WINDOW_NORMAL)
cv2.createTrackbar('kernel','binary',1,100,kernelChange)#确定为6
cv2.imshow('binary',binary)
cv2.waitKey()
# 销毁窗口
cv2.destroyAllWindows()

