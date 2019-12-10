import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

# def metric(fn):
#     print('%s executed in %s ms' % (fn.__name__, 10.24))
#     return fn
# @metric
# def customize(a,b):
#     print("我是被装饰的函数,运行结果%d"%(a+b))
# customize(1,3)
def blursImg(imgInfoList):
    plt.figure()
    i=0
    for imginfo in imgInfoList:
        blurName,imgData,dealTime=imginfo
        i=i+1
        plt.subplot(1,2,i)
        plt.imshow(imgData)
        plt.xticks([])
        plt.yticks([])
        tempyime="%0.3f" % dealTime
        plt.title("{0}({1}ms)".format(blurName,str(tempyime)), fontsize=20)
    plt.show()

import datetime
#添加处理时间信息
def dealTime(fun,**kwargs):
    start=datetime.datetime.now()
    imgData=fun(**kwargs)
    end=datetime.datetime.now()
    totalMs=(end-start).total_seconds()
    ret=[imgData,totalMs*1000]
    return ret

def HoughCircles(img,param1,param2):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=param1, param2=param2, minRadius=700, maxRadius=840)
    circles = np.uint16(np.around(circles))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), thickness=6)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 3)
    return img

def AreaCircles(img):
    binaryImg=cv2.imread('OTSU_binary.png',cv2.IMREAD_GRAYSCALE)
    _, contours, _ = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        print(area)
        if (area < 2000000):
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        ratio = 0.0
        if (y != 0):
            ratio = float(w / h)
        if ((ratio > 0.9) & (ratio < 1.1)):
            cv2.drawContours(img, contours, i, (255, 0, 0),thickness=6)
            size = 100
            # src = src[y - size:y + h + size, x - size:x + w + size]
    # cv2.imshow('result_img',img)
    return img

img=cv2.imread('09_30_13.jpg')
AreaImgInfo=dealTime(AreaCircles,img=img.copy())
HoughImgInfo=dealTime(HoughCircles,img=img.copy(),param1=70,param2=80)
nameList=['面积法','Hough圆检测法']
imgInfoList=[AreaImgInfo,HoughImgInfo]
#将信息添加到数组里面
fullImgInfoList=[]
for i in range(len(nameList)):
    imgData,time=imgInfoList[i]
    name=nameList[i]
    imgFullInfo=[name,imgData,time]
    fullImgInfoList.append(imgFullInfo)
#运行显示图片
blursImg(fullImgInfoList)

