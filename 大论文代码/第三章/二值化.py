import cv2 as cv
def threshold_OTSU(image):
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)   #要二值化图像，要先进行灰度化处理
    ret, binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value: %s"%ret)#打印阈值，前面先进行了灰度处理0-255，我们使用该阈值进行处理，低于该阈值的图像部分全为黑，高于该阈值则为白色
    cv.imshow("binary",binary)#显示二值化图像
    cv.waitKey()

def local_threshold(image):
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)   #要二值化图像，要先进行灰度化处理
    dst = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,25,10)
    cv.imshow("local_threshold", dst);cv.waitKey()

def myowm_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # 要二值化图像，要先进行灰度化处理
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY )
    print("threshold value: %s" % ret)  # 打印阈值，前面先进行了灰度处理0-255，我们使用该阈值进行处理，低于该阈值的图像部分全为黑，高于该阈值则为白色
    cv.imshow("binary", binary)  # 显示二值化图像
    cv.waitKey()

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
def thresholdImg(imgInfoList):
    plt.figure()
    i=0
    for imginfo in imgInfoList:
        blurName,imgData,dealTime=imginfo
        i=i+1
        plt.subplot(1,3,i)
        plt.imshow(imgData,cmap='gray')
        plt.xticks([])
        plt.yticks([])
        tempyime="%0.3f" % dealTime
        plt.title("{0}({1}ms)".format(blurName,str(tempyime)), fontsize=10)
    plt.show()

import datetime
#添加处理时间信息
def dealTime(fun,**kwargs):
    start=datetime.datetime.now()
    _,imgData=fun(**kwargs)
    end=datetime.datetime.now()
    totalMs=(end-start).total_seconds()
    ret=[imgData,totalMs*1000]
    return ret

sourceImg=cv.imread('09_30_13.jpg',cv.IMREAD_ANYCOLOR)
# sourceImg = cv.cvtColor(sourceImg, cv.COLOR_BGR2GRAY)
sourceImgInfo=[sourceImg,0.0]
#OTSU
OTSU_thresh=dealTime(cv.threshold,src=sourceImg,thresh=0,maxval=255,type=cv.THRESH_BINARY | cv.THRESH_OTSU)
#全局自适应
global_thresh=dealTime(cv.threshold,src=sourceImg,thresh=0,maxval=255,type=cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
nameList=['原图','OTSU','全局自适应']
imgInfoList=[sourceImgInfo,OTSU_thresh,global_thresh]
fullImgInfoList=[]
#将信息添加到数组里面
for i in range(len(nameList)):
    imgData,time=imgInfoList[i]
    name=nameList[i]
    imgFullInfo=[name,imgData,time]
    fullImgInfoList.append(imgFullInfo)
#运行显示图片
thresholdImg(fullImgInfoList)
