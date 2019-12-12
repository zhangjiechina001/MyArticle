import math

import cv2
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

import datetime
#添加处理时间信息
def dealTime(fun,**kwargs):
    start=datetime.datetime.now()
    imgData=fun(**kwargs)
    end=datetime.datetime.now()
    totalMs=(end-start).total_seconds()
    ret=[imgData,totalMs*1000]
    return ret

def calcMax(img):
    h,w,_=img.shape
    kernel=np.ones((h,220),dtype=np.float32)
    # kernel[0:10,:]=0
    result = signal.convolve2d(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), kernel, 'valid')
    hist=result.ravel()
    ret_max=np.max(hist)
    start_position=np.argmax(hist)
    # start_position=start_position[0]
    ret_theta=(start_position/w)*2*np.pi
    return ret_max,ret_theta

def unFlodImgPro(img,startTheta,endTheta):# 得到圆形区域的中心坐标
    x0 = img.shape[0] // 2
    y0 = img.shape[1] // 2
    unwrapped_height = radius = img.shape[0] // 2  # 展开后的高
    full_width= int(2 * math.pi * radius)   #总长
    unwrapped_width = int(2 * math.pi * radius*(endTheta-startTheta)/(2*math.pi))  # 展开后的宽
    unwrapped_img = np.zeros((unwrapped_height-850, unwrapped_width, 3), dtype="u1")
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
    unwrapped_img=unwrapped_img[8:unwrapped_height-1,:,:]
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
        ret_info=img2_info[0],img2_info[1]-0.5*math.pi

    max,theta=ret_info
    # theta=theta-0.5
    ret_img=unFlodImgPro(src,startTheta=-theta,endTheta=-theta+0.95)
    return ret_img

def last_fun(img):
    #先对原图进行两次金字塔处理=>展开小图=>两次卷积取得最佳数据=>展开大图
    src=cv2.pyrDown(img)
    src=cv2.pyrDown(src)
    img_unflood1=unflood(src,0,2*math.pi)
    img_unflood2 = unflood(src, 0.5 * math.pi, 2.5 * math.pi)
    cv2.namedWindow('img_unflood1',cv2.WINDOW_NORMAL)
    cv2.namedWindow('img_unflood2', cv2.WINDOW_NORMAL)
    # plt.imshow(img_unflood1)
    # plt.show()
    cv2.imshow('img_unflood1', img_unflood1)
    cv2.imshow('img_unflood2', img_unflood2)
    img_info1=calcMax(img_unflood1)
    img_info2=calcMax(img_unflood2)

    ret_img=unflood_imgPro(img,img_info1,img_info2)
    ret_img=ret_img[20:,:,:]
    return ret_img





src=cv2.imread('unflood_binary.png')
# src=binary_img(src)
img_info=dealTime(last_fun,img=src)
print('{0}ms'.format(str(img_info[1])))
# img=last_fun(src)
# cv2.putText(img, '000', (50, 300), font, 1.2, (255, 255, 255), 2)
font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img_info[0],"this is flower ",(0,30),font,1,(200,100,255),3,cv2.LINE_AA)
# cv2.putText(img_info[0],str(img_info[1]),(50, 300), cv2.FONT_HERSHEY_COMPLEX, 5.0, (255, 255, 255), 2)
cv2.imshow('last_img',img_info[0])

cv2.waitKey()
