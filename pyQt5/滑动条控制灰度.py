# -*- coding: utf-8 -*-

import cv2
import numpy as np


img = cv2.imread('09_30_13.jpg')
# img=cv2.pyrDown(img)
zeros=np.zeros(shape=img.shape,dtype=np.uint8)
def callback(x):
    gamma=x/255
    global img
    global zeros
    src,img2 = cv2.threshold(img,x,155,cv2.THRESH_BINARY)
    cv2.imshow('image',img2)

# 创建一副黑色图像
# 设置滑动条组件
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.createTrackbar('R', 'image', 0, 255, callback)
cv2.createTrackbar('G', 'image', 0, 255, callback)
cv2.createTrackbar('B', 'image', 0, 255, callback)
cv2.waitKey()

# 销毁窗口
cv2.destroyAllWindows()

