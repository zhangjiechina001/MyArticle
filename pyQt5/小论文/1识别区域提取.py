'''先找到感兴趣圆环，再将矩形切割为正方形，最后根据对图片进行卷积，裁剪出感兴趣区域'''
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
#1.先找到感兴趣圆环区域
#读取图片
src=cv2.imread('09_30_13.jpg',cv2.IMREAD_GRAYSCALE)

#二值化处理
ret,binary=cv2.threshold(src,87,255,cv2.THRESH_BINARY)
cv2.namedWindow('binary',cv2.WINDOW_NORMAL)
cv2.imshow('binary1',binary)

#形态学处理
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#CLOSE先膨胀再腐蚀
dst_img=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
#OPEN先腐蚀再膨胀
dst_img=cv2.morphologyEx(dst_img,cv2.MORPH_OPEN,kernel)
cv2.imshow('morpho',dst_img)

#寻找边界
#返回的是原图片，边界集合，轮廓的属性
src=dst_img
_,contours,_=cv2.findContours(dst_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
result_img=np.zeros(src.shape,np.uint8)
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
        cv2.drawContours(result_img,contours,i,(255,255,255))
        size=100
        src = cv2.imread('09_30_40.jpg', cv2.IMREAD_GRAYSCALE)
        src=src[y-size:y+h+size,x-size:x+w+size]
cv2.imshow('contours',result_img)
cv2.imwrite('contours.png',result_img)
cv2.namedWindow('result',cv2.WINDOW_NORMAL)
cv2.imshow('result',src)

#圆环拉升为矩形
#找到圆形区域的中心坐标
x0=src.shape[0]//2
y0=src.shape[1]//2
#通过圆形区域半径展开图像
unwrapped_height = radius = src.shape[0] // 2#展开后的高
unwrapped_width = int(2 * math.pi * radius)#展开后的宽
unwrapped_img = np.zeros((unwrapped_height, unwrapped_width), dtype="u1")
except_count = 0
for j in range(unwrapped_width):
    theta = -2 * math.pi * (j / unwrapped_width)  # 1. start position such as "+ math.pi"
    # theta=theta+0.75*math.pi
    for i in range(unwrapped_height-850):
        unwrapped_radius = radius - i  # 2. don't forget
        x = unwrapped_radius * math.cos(theta) + x0  # 3. "sin" is clockwise but "cos" is anticlockwise
        y = unwrapped_radius * math.sin(theta) + y0
        x, y = int(x), int(y)
        try:
            unwrapped_img[i, j] = src[x, y]
        except Exception as e:
            except_count = except_count + 1
print('except_count:'+str(except_count))
# cv2.namedWindow('Unwrapped',cv2.WINDOW_NORMAL)
unwrapped_img=unwrapped_img[20:80,:]
# src=cv2.resize(src,(30,30))
plt.imshow(cv2.cvtColor(unwrapped_img,cv2.COLOR_GRAY2RGB))#1944
plt.xlabel('w',fontproperties='SimHei',fontsize=15)
plt.ylabel('h',fontproperties='SimHei',fontsize=15)
plt.minorticks_on()
plt.xticks([])
plt.yticks([])
plt.show()
# cv2.imshow("Unwrapped", unwrapped_img)
# cv2.imwrite('unFlood.jpg',unwrapped_img)

#对原图进行二值化，去燥，感兴趣区截取操作
_,binary=cv2.threshold(unwrapped_img,145,255,cv2.THRESH_BINARY)
cv2.imshow("binary", binary)
cv2.imwrite("binary.jpg", binary)
#形态学处理
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#CLOSE先膨胀再腐蚀
dst_img=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
#OPEN先腐蚀再膨胀
dst_img=cv2.morphologyEx(dst_img,cv2.MORPH_OPEN,kernel)
cv2.namedWindow('morpho',cv2.WINDOW_NORMAL)
cv2.imshow('morpho',dst_img)

#对原图片进行卷积，得出定位点
h,w=dst_img.shape
src=cv2.bitwise_not(dst_img)
kernel=np.ones((h,850),dtype=np.float32)
result=signal.convolve2d(src,kernel,'valid')
hist=result.ravel()
plt.plot(hist)
maxIdx=np.argmax(hist)
cutimg=dst_img[:,maxIdx:maxIdx+860]
cv2.imwrite('afterCutImg.png',cutimg)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()