'''先找到感兴趣圆环，再将矩形切割为正方形，最后根据对图片进行卷积，裁剪出感兴趣区域'''
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
#1.先找到感兴趣圆环区域
#读取图片
#输入灰度图，返回二值化后的图
def _binaryAndmorphologEx(src):
    ret, binary = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    #形态学处理
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # CLOSE先膨胀再腐蚀
    dst_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # OPEN先腐蚀再膨胀
    dst_img = cv2.morphologyEx(dst_img, cv2.MORPH_OPEN, kernel)
    cv2.imshow('morpho', dst_img)
    return dst_img

#分割图，先对二值图进行特征特区，然后在对缘儒进行处理,返回图像的圆心，半径
def _cutImg(src,readImg):
    _, contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result_img = np.zeros(src.shape, np.uint8)
    retImg=None
    i=1
    circle_point = None
    circle_r = None
    # for contour in contours:
    #     area=cv2.contourArea(contour)
    #     print('area{0}:{1}'.format(i,str(area)))
    #     i+=1

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        print(area)
        if (area < 1500000 or area>2500000):
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        ratio = 0.0
        if (y != 0):
            ratio = float(w / h)
        if ((ratio > 0.95) & (ratio < 1.05)):
            cv2.drawContours(result_img, contours, i, (255, 255, 255))
            cv2.namedWindow('cut',cv2.WINDOW_NORMAL)
            cv2.imshow('cut',result_img)
            # if(area>2000000 and area<2500000):
            size = 100
            # else:
            #     size=0
            # retImg = readImg[y - size:y + h + size, x - size:x + w + size]
            circle_point=(y+w//2,x+h//2)
            circle_r=w//2
            break
    return circle_point,circle_r

def dealImg(img):
    tempimg=img.copy()
    binary=_binaryAndmorphologEx(tempimg)
    point,r=_cutImg(binary,img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    unflood_img=unfloodImage(img,point,r,150)
    cv2.namedWindow('After cut image',cv2.WINDOW_NORMAL)
    cv2.imshow('After cut image',unflood_img)
    cv2.waitKey()
    return unflood_img

#输入图片，圆心点，展开内半径，展开宽度
def unfloodImage(img,point,unfloodR,width):
    h,w,_=img.shape
    x0,y0=point
    unwrapped_width=unfloodR+width#展开的最大半径
    unwrapped_height=width
    full_width=int(2*math.pi*unwrapped_width)#展开后的长度
    unwrapped_img=np.zeros((unwrapped_height,full_width,3),dtype='u1')
    except_count=0
    for j in range(full_width):
        theta = -2 * math.pi * (j / full_width)  # 1. 开始位置
        # theta=theta+0.75*math.pi
        for i in range(unwrapped_height):
            unwrapped_radius = unwrapped_width -i  # 2. don't forget
            x = unwrapped_radius * math.cos(theta) + x0  #
            y = unwrapped_radius * math.sin(theta) + y0
            x, y = int(x), int(y)
            try:
                if x<0 or x>=h or y<0 or y>=w:
                    continue
                unwrapped_img[i, j, :] = img[x, y, :]
            except Exception as e:
                except_count = except_count + 1
    print('expect count:'+str(except_count))
    return unwrapped_img

img=cv2.imread('OKPictures//17_34_06.jpg',cv2.IMREAD_GRAYSCALE)
img=dealImg(img)
# img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
# ret_img=unfloodImage(img,(899,899),800,200)
# import 两次卷积处理错分割问题 as unflood
# ret_img=unflood.binary_img(img)
# cv2.imshow('binary image',ret_img)
# ret_img=cv2.cvtColor(ret_img,cv2.COLOR_GRAY2BGR)
# ret_img=unflood.last_fun(ret_img)
# cv2.namedWindow('last_img',cv2.WINDOW_NORMAL)
#
# cv2.imshow('last_img',ret_img)
cv2.waitKey()
