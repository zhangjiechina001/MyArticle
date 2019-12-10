import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

def blursImg(imgInfoList):
    plt.figure()
    i=0
    for imginfo in imgInfoList:
        blurName,imgData,dealTime=imginfo
        # imgData = cv2.cvtColor(imgData, cv2.COLOR_GRAY2BGR)
        i=i+1
        plt.subplot(2,3,i)
        plt.imshow(imgData)
        plt.xticks([])
        plt.yticks([])
        tempyime="%0.3f" % dealTime
        plt.title("{0}({1}ms)".format(blurName,str(tempyime)), fontsize=10)
        # cv2.imshow(blurName,imgData)
    plt.show()

# def metric(fn):
#     print('%s executed in %s ms' % (fn.__name__, 10.24))
#     return fn
# @metric
# def customize(a,b):
#     print("我是被装饰的函数,运行结果%d"%(a+b))
# customize(1,3)
import datetime
#添加处理时间信息
def dealTime(fun,**kwargs):
    start=datetime.datetime.now()
    imgData=fun(**kwargs)
    end=datetime.datetime.now()
    totalMs=(end-start).total_seconds()
    ret=[imgData,totalMs*1000]
    return ret

# 第六个参数param1是Canny边缘检测的高阈值，低阈值被自动置为高阈值的一半；
#
# 第七个参数param2是累加平面对是否是圆的判定阈值；
def HoughCircles(img,param1,param2):
    # garry=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
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


#进行五种不同阀值的Hough图片的处理


sourceImg = cv2.imread('09_30_13.jpg',0)
sourceImg = cv2.medianBlur(sourceImg,5)
sourceImgInfo=[sourceImg,0.0]
inputParams=[[50,100],[60,90],[70,80],[50,90],[50,80],[50,70]]
imgInfoList=[]
for para in inputParams:
    imgInfo=dealTime(HoughCircles,img=sourceImg.copy(),param1=para[0],param2=para[1])
    imgInfoList.append(imgInfo)
nameList=['param1=50,param2=100','param1=60,param2=90','param1=70,param2=80','param1=50,param2=90','param1=50,param2=80','param1=50,param2=70']

fullImgInfoList=[]
#将信息添加到数组里面
for i in range(len(nameList)):
    imgData,time=imgInfoList[i]
    name=nameList[i]
    imgFullInfo=[name,imgData,time]
    fullImgInfoList.append(imgFullInfo)

#运行显示图片
blursImg(fullImgInfoList)