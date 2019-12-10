import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标

def showImgInfo(imgInfoList):
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

def drawHist(img):
    #encoding:utf-8
    src = cv2.imread('test01.jpg')
    cv2.imshow("src", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.hist(src.ravel(), 256)
    plt.show()

src=cv2.imread('')