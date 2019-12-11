import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

def blursImg(imgInfoList):
    plt.figure()
    i=0
    for imginfo in imgInfoList:
        blurName,imgData,dealTime=imginfo
        i=i+1
        plt.subplot(2,3,i)
        plt.imshow(imgData)
        # plt.xticks([])
        # plt.yticks([])
        plt.xlabel('x轴')
        plt.ylabel('y轴',fontsize=20)
        tempyime="%0.3f" % dealTime
        plt.title("{0}({1}ms)".format(blurName,str(tempyime)), fontsize=10)
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

def dealCornerBlur(imgSource,cornerImg):
    dst = cv2.dilate(cornerImg, None)  # 图像膨胀
    # Threshold for an optimal value, it may vary depending on the image.
    # print(dst)
    # img[dst>0.00000001*dst.max()]=[0,0,255] #可以试试这个参数，角点被标记的多余了一些
    imgSource[dst > 0.01 * dst.max()] = [255, 0, 0]  # 角点位置用红色标记
    return imgSource


#进行五种图片的处理
sourceImg=cv2.imread('blurImg.png',cv2.IMREAD_ANYCOLOR)
sourceImgInfo=[sourceImg,0.0]
#均值滤波
meanBlur=dealTime(cv2.blur,src=sourceImg,ksize=(5,5))
#高斯滤波
gaussBlur=dealTime(cv2.GaussianBlur,src=sourceImg,ksize=(7,7),sigmaX=0)
#双线性
doubleLineBlur=dealTime(cv2.bilateralFilter,src=sourceImg, d=20, sigmaColor=50, sigmaSpace=15)
#Sobel
sobelBlur=dealTime(cv2.Sobel,src=sourceImg,ddepth=cv2.CV_16S,dx=1,dy=1)
#角点
cornerBlur=dealTime(cv2.cornerHarris,src=cv2.cvtColor(src=sourceImg,code=cv2.COLOR_BGR2GRAY),blockSize=2,ksize=3,k=0.04)
cornerBlur[0]=dealCornerBlur(sourceImg.copy(),cornerBlur[0])

nameList=['原图','均值滤波','高斯滤波','双边滤波','Sobel滤波','角点滤波']
imgInfoList=[sourceImgInfo,meanBlur,gaussBlur,doubleLineBlur,sobelBlur,cornerBlur]
fullImgInfoList=[]
#将信息添加到数组里面
for i in range(len(nameList)):
    imgData,time=imgInfoList[i]
    name=nameList[i]
    imgFullInfo=[name,imgData,time]
    fullImgInfoList.append(imgFullInfo)
#运行显示图片
blursImg(fullImgInfoList)