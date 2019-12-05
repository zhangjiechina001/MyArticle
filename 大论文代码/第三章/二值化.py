import cv2 as cv
import datetime
def threshold_OTSU(image):
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)   #要二值化图像，要先进行灰度化处理
    ret, binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value: %s"%ret)#打印阈值，前面先进行了灰度处理0-255，我们使用该阈值进行处理，低于该阈值的图像部分全为黑，高于该阈值则为白色
    cv.imshow("binary",binary)#显示二值化图像
    cv.waitKey()

def local_gaussthreshold(image):
    gray = image
    start = datetime.datetime.now()
    dst = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,41,10)
    end = datetime.datetime.now()
    totalMs=(end-start).total_seconds()
    return [dst,totalMs*1000]

def local_meanthreshold(image):
    gray = image
    start = datetime.datetime.now()
    dst = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,25,10)
    end = datetime.datetime.now()
    totalMs=(end-start).total_seconds()
    return [dst,totalMs*1000]

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
        plt.subplot(2,3,i)
        plt.imshow(imgData,cmap='gray')
        plt.xticks([])
        plt.yticks([])
        tempyime="%0.3f" % dealTime
        plt.title("{0}({1}ms)".format(blurName,str(tempyime)), fontsize=20)
    plt.show()


#添加处理时间信息
def dealTime(fun,**kwargs):
    start=datetime.datetime.now()
    threshValue,imgData=fun(**kwargs)
    end=datetime.datetime.now()
    totalMs=(end-start).total_seconds()
    ret=[imgData,totalMs*1000]
    print(str(threshValue))
    return ret

sourceImg=cv.imread('bestimg.jpg',cv.IMREAD_GRAYSCALE)
# sourceImg = cv.cvtColor(sourceImg, cv.COLOR_BGR2GRAY)
sourceImgInfo=[sourceImg,0.0]
#OTSU
OTSU_thresh=dealTime(cv.threshold,src=sourceImg,thresh=0,maxval=255,type=cv.THRESH_BINARY | cv.THRESH_OTSU)
#全局自适应
global_thresh=dealTime(cv.threshold,src=sourceImg,thresh=0,maxval=255,type=cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
#局部高斯平均
local_thresh=local_gaussthreshold(sourceImg)
#局部平均
local_meanthresh=local_gaussthreshold(sourceImg)
#自定义阀值
userDefine_thresh=dealTime(cv.threshold,src=sourceImg,thresh=138,maxval=255,type=cv.THRESH_BINARY)
nameList=['原图','OTSU','全局自适应','局部高斯自适应','局部平均自适应','人工选择']
imgInfoList=[sourceImgInfo,OTSU_thresh,global_thresh,local_thresh,local_meanthresh,userDefine_thresh]
fullImgInfoList=[]
#将信息添加到数组里面
for i in range(len(nameList)):
    imgData,time=imgInfoList[i]
    name=nameList[i]
    imgFullInfo=[name,imgData,time]
    fullImgInfoList.append(imgFullInfo)
#运行显示图片
thresholdImg(fullImgInfoList)
