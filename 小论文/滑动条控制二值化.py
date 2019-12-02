import cv2
import numpy as np

img = cv2.imread("09_30_13.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img=cv2.pyrDown(img)
# img=cv2.equalizeHist(img)
# img=cv2.Canny(img,100,200)
cv2.imshow('img',img)
def callBack(x):
    global img
    global binary
    src,binary=cv2.threshold(img,x,255,cv2.THRESH_BINARY)
    cv2.imshow('binary',binary)

cv2.namedWindow('binary')
cv2.createTrackbar('reszieThreshold','binary',100,255,callBack)
cv2.waitKey()
cmd=input()
if cmd=='save':
    cv2.imwrite('bestimg.jpg',binary)
    print('save susessful!')