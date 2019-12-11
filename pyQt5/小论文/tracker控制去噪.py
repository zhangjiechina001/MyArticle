import cv2

import cv2
import numpy as np

img = cv2.imread("unflood_binary.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('img',img[13:44,1000:1700])
def callBack(x):
    global img
    global binary
    binary=img
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(x,x))
    # binary=cv2.dilate(binary,kernel)
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
    binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
    # src,binary=cv2.threshold(img,x,255,cv2.THRESH_BINARY)
    cv2.imshow('binary',binary)

cv2.namedWindow('binary',cv2.WINDOW_NORMAL)
cv2.createTrackbar('reszieThreshold','binary',0,10,callBack)
cv2.waitKey()
cmd=input()
if cmd=='save':
    cv2.imwrite('bestimg.jpg',binary)
    print('save susessful!')
