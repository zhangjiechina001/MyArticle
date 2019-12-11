import cv2
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
#读取图片
src=cv2.imread('unFloodAfterBinary.jpg',cv2.IMREAD_GRAYSCALE)
bit_wise=cv2.bitwise_not(src)
cv2.namedWindow('src',cv2.WINDOW_NORMAL)

def distributeImg(imgInfoList):
    plt.figure()
    i=0
    for imginfo in imgInfoList:
        blurName,imgData,dealTime=imginfo
        i=i+1
        plt.subplot(3,1,i)
        if(blurName=='原图'):
            # cv2.cvtColor(imgData,cv2.COLOR_RGB2GRAY)
            plt.imshow(imgData,cmap='gray')
        else:
            plt.plot(imgData,color='r')
        # plt.xticks([])
        # plt.yticks([])
        # plt.xlabel('x轴')
        # plt.ylabel('y轴',fontsize=20)
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

#卷积法计算投影分布
def projectionDeistributionConvolve(img):
    h, w = img.shape
    kernel=np.ones((h,1),dtype=np.float32)
    result = signal.convolve2d(img, kernel, 'valid')
    hist = result.ravel()
    return hist

#遍历法计算投影分布
def projectionDestributionErgodic(img):
    h,w=img.shape
    ret=[]
    #先计算每列，在计算每一行
    for i in range(w):
        hDestr=0
        for j in range(h):
            if(img[j,i]==0):
                continue
            hDestr+=img[j,i]
        ret.append(hDestr)
    return ret


Ergofic=dealTime(projectionDestributionErgodic,img=bit_wise.copy())
convolve= dealTime(projectionDeistributionConvolve,img=bit_wise.copy())
sorImg=[bit_wise,0]
nameList=['原图','遍历算法','卷积算法']
imgInfoList=[sorImg,Ergofic,convolve]
fullImgInfoList=[]
#将信息添加到数组里面
for i in range(len(nameList)):
    imgData,time=imgInfoList[i]
    name=nameList[i]
    imgFullInfo=[name,imgData,time]
    fullImgInfoList.append(imgFullInfo)

distributeImg(fullImgInfoList)
cv2.imshow('src',bit_wise)
cv2.waitKey()