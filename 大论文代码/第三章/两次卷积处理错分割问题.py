import math

import cv2
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

def calcMax(img):
    h,w,_=img.shape
    kernel=np.ones((h,200),dtype=np.float32)
    # kernel[0:10,:]=0
    result = signal.convolve2d(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), kernel, 'valid')
    hist=result.ravel()
    ret_max=np.max(hist)
    start_position=np.argmax(hist)
    # start_position=start_position[0]
    ret_theta=(start_position/w)*2*np.pi-np.pi
    return ret_max,ret_theta

def unFlodImgPro(img,startTheta,endTheta):# 得到圆形区域的中心坐标
    x0 = img.shape[0] // 2
    y0 = img.shape[1] // 2
    unwrapped_height = radius = img.shape[0] // 2  # 展开后的高
    full_width= int(2 * math.pi * radius)   #总长
    unwrapped_width = int(2 * math.pi * radius*(endTheta-startTheta)/(2*math.pi))  # 展开后的宽
    unwrapped_img = np.zeros((unwrapped_height, unwrapped_width, 3), dtype="u1")
    except_count = 0
    for j in range(unwrapped_width):
        theta = -2 * math.pi * (j / full_width)+startTheta  # 1. 开始位置
        # theta=theta+0.75*math.pi
        for i in range(unwrapped_height-850):
            unwrapped_radius = radius - i  # 2. don't forget
            x = unwrapped_radius * math.cos(theta) + x0  # 3. "sin" is clockwise but "cos" is anticlockwise
            y = unwrapped_radius * math.sin(theta) + y0
            x, y = int(x), int(y)
            try:
                unwrapped_img[i, j, :] = img[x, y, :]
            except Exception as e:
                except_count = except_count + 1
    print(except_count)
    return unwrapped_img

#展开小图
def unflood(img,startTheta,endTheta):# 得到圆形区域的中心坐标
    x0 = img.shape[0] // 2
    y0 = img.shape[1] // 2
    unwrapped_height = radius = img.shape[0] // 2  # 展开后的高
    full_width= int(2 * math.pi * radius)   #总长
    unwrapped_width = int(2 * math.pi * radius*(endTheta-startTheta)/(2*math.pi))  # 展开后的宽
    unwrapped_img = np.zeros((unwrapped_height-850//4, unwrapped_width,3), dtype="u1")
    except_count = 0
    for j in range(unwrapped_width):
        theta = -2 * math.pi * (j / full_width)+startTheta  # 1. 开始位置
        # theta=theta+0.75*math.pi
        for i in range(unwrapped_height-850//4):
            unwrapped_radius = radius - i  # 2. don't forget
            x = unwrapped_radius * math.cos(theta) + x0  # 3. "sin" is clockwise but "cos" is anticlockwise
            y = unwrapped_radius * math.sin(theta) + y0
            x, y = int(x), int(y)
            try:
                unwrapped_img[i, j, :] = img[x, y, :]
            except Exception as e:
                except_count = except_count + 1
    print(except_count)
    # unwrapped_img=unwrapped_img[10:unwrapped_height-1,:,:]
    return unwrapped_img

def binary_img(img):
    _,ret_img=cv2.threshold(img,141,255,cv2.THRESH_BINARY_INV)
    return ret_img

def unflood_imgPro(src,img1_info,img2_info):
    ret_info=None
    #如果第一个大于等于第二个，那就第一个，否则选第二组数据
    if(img1_info[0]>=img2_info[0]):
        ret_info=img1_info
    else:
        ret_info=img2_info

    ret_img=unFlodImgPro(src,startTheta=1.16*math.pi,endTheta=1.66*math.pi)
    return ret_img

def last_fun(img):
    #先对原图进行两次金字塔处理=>展开小图=>两次卷积取得最佳数据=>展开大图
    src=cv2.pyrDown(img)
    src=cv2.pyrDown(src)
    img_unflood1=unflood(src,0,2*math.pi)
    img_unflood2 = unflood(src, 0.5 * math.pi, 2.5 * math.pi)
    cv2.namedWindow('img_unflood1',cv2.WINDOW_NORMAL)
    cv2.namedWindow('img_unflood2', cv2.WINDOW_NORMAL)
    plt.imshow(img_unflood1)
    # plt.show()
    # cv2.imshow('img_unflood1', img_unflood1)
    # cv2.imshow('img_unflood2', img_unflood2)
    img_info1=calcMax(img_unflood1)
    img_info2=calcMax(img_unflood2)
    # hist1, ret_max1, ret_theta1=img_info1
    # hist, ret_max, ret_theta=img_info2
    # plt.subplot(2,2,1)
    # plt.imshow(img_unflood1)
    # plt.subplot(2,2,2)
    # plt.plot(hist1)
    # plt.subplot(2,2,3)
    # plt.imshow(img_unflood2)
    # plt.subplot(2,2,4)
    # plt.plot(hist)
    # plt.show()

    ret_img=unflood_imgPro(img,img_info1,img_info2)
    return ret_img





src=cv2.imread('unflood_binary.png')
img=last_fun(src)
cv2.imshow('last_img',img)
# src=cv2.pyrDown(src)
# src=cv2.pyrDown(src)
# unflood_img=unflood(src,startTheta=0,endTheta=2*math.pi)
# binary_img=binary_img(src)
# cv2.namedWindow('unflood_img',cv2.WINDOW_NORMAL)
# cv2.imshow('unflood_img',binary_img)
# cv2.imwrite('unflood_img_binary.png',binary_img)
cv2.waitKey()
# src=cv2.imread('imgSave.jpg',cv2.IMREAD_GRAYSCALE)
# bit_wise=cv2.bitwise_not(src)
# max,theta=calcMax(bit_wise)
# cv2.imshow('pic',src)