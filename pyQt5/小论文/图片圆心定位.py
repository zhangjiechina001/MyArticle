import cv2
import matplotlib.pyplot as plt
import numpy as np


src=cv2.imread('dealed.png',cv2.IMREAD_GRAYSCALE)
w,h=src.shape
# src=cv2.cvtColor(src,cv2.COLOR_GRAY2RGB)
_,binary=cv2.threshold(src,142,255,cv2.THRESH_BINARY)
tempsrc=src
#寻找边界
#返回的是原图片，边界集合，轮廓的属性
_,contours,_=cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
result_img=np.zeros(src.shape,np.uint8)
cv2.imshow('binary',binary)
for i in range(len(contours)):
    area=cv2.contourArea(contours[i])
    print(area)
    if(area<2000000):
        continue
    x,y,w,h=cv2.boundingRect(contours[i])
    ratio=0.0
    if(y!=0):
        ratio=float(w/h)
    if((ratio>0.9)&(ratio<1.1)):
        cv2.drawContours(result_img,contours,i,(255,255,255),thickness=15)
        size=100
        src=src[y-size:y+h+size,x-size:x+w+size]
cv2.imshow('contours',result_img)
cv2.imshow('result',src)


#二值图
plt.subplot2grid(shape=(3,3),loc=(0,0))
plt.imshow(cv2.cvtColor(tempsrc,cv2.COLOR_GRAY2RGB))
plt.xlabel('(a)原图',fontproperties='SimHei',fontsize=15)
plt.xticks([])
plt.yticks([])
plt.subplot2grid(shape=(3,3),loc=(0,1))
plt.imshow(cv2.cvtColor(result_img,cv2.COLOR_GRAY2RGB))
plt.xlabel('(b)圆心定位',fontproperties='SimHei',fontsize=15)
# plt.
plt.xticks([])
plt.yticks([])
plt.subplot2grid(shape=(3,3),loc=(0,2))
# src=cv2.resize(src,(30,30))
plt.imshow(cv2.cvtColor(src,cv2.COLOR_GRAY2RGB))#1944
plt.xlabel('(c)最小区域',fontproperties='SimHei',fontsize=15)
plt.minorticks_on()
plt.xticks([])
plt.yticks([])
plt.show()