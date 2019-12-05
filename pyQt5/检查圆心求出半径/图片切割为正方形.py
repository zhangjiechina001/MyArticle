import cv2
import numpy as np

src=cv2.imread('09_30_40.jpg')
cv2.imshow('original',src)
#图像二值化
_,binary=cv2.threshold(src,87,255,type=cv2.THRESH_BINARY)
cv2.namedWindow('binary',cv2.WINDOW_NORMAL)
cv2.imshow('binary',binary)
#找出图像最内径圆心和半径
cv2.waitKey()
cv2.destroyAllWindows()
