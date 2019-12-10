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

def drawHist(imgInfoList):
    #encoding:utf-8
    i=0
    for imgInfo in imgInfoList:
        img_name=imgInfo[0]
        img_data=imgInfo[1]
        i=i+1
        plt.subplot(2,3,i)
        thresh, mask_img = cv2.threshold(src=img_data, thresh=73, maxval=255, type=cv2.THRESH_BINARY)
        hsit = cv2.calcHist([img_data], [0], mask_img, [256], [0, 255])
        plt.plot(hsit,color='r')
        cv2.imshow(img_name,mask_img)
        plt.title("{0}".format(img_name), fontsize=20)

    plt.show()

ok_src1=cv2.imread('OKPictures//16_24_38.jpg')
ok_src2=cv2.imread('OKPictures//16_50_50.jpg')
ok_src3=cv2.imread('OKPictures//16_51_22.jpg')
ng_src1=cv2.imread('NGPictures//16_16_08.jpg')
ng_src2=cv2.imread('NGPictures//16_16_11.jpg')
ng_src3=cv2.imread('NGPictures//16_16_57.jpg')
imgData_list=[ok_src1,ok_src2,ok_src3,ng_src1,ng_src2,ng_src3]
name_list=['ok1','ok2','ok3','ng1','ng2','ng3']
imgFullInfo_list=[]
for i in range(len(name_list)):
    imgData_list[i]=cv2.cvtColor(imgData_list[i],cv2.COLOR_RGB2GRAY)
    imgFullInfo_list.append([name_list[i],imgData_list[i]])

drawHist(imgFullInfo_list)

