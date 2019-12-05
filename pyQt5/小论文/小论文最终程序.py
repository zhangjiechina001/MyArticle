import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
#识别流程为1.读取照片=>2.图像预处理（二值化,形态学处理）=>3.寻找最大区域（确定工件中心位置）=>4.极坐标展开
#=>5.卷积处理，得到字符分布特征=>6.字符分割

#region 图像预处理
#二值化，去噪处理
def ImagePorcessing(img):
    _,result=cv2.threshold(img,142,255,cv2.THRESH_BINARY)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    result=cv2.morphologyEx(result,cv2.MORPH_CLOSE,kernel)
    result=cv2.morphologyEx(result,cv2.MORPH_OPEN,kernel)
    return result

def FindMaxArea(img):
    #寻找边界
    _,contours,_=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #寻找合适区域
    for i in range(len(contours)):
        area=cv2.contourArea(contours[i])
        if(area<2000000):
            continue
        x,y,w,h=cv2.boundingRect(contours[i])
        ratio=0.0
        if(y!=0):
            ratio=float(w/h)
        if(ratio>0.9)&(ratio<1.1):
            size=80
            result=img[y-size:y+h+size,x-size:x+w+size]

    return  result

def ExpandImg(img):
    import math
    # 圆环拉升为矩形
    # 找到圆形区域的中心坐标
    x0=img.shape[0]//2
    y0=img.shape[1]//2
    # 通过圆形区域半径展开图像
    unwrapped_height = radius = img.shape[0] // 2  # 展开后的高
    unwrapped_width = int(2 * math.pi * radius)  # 展开后的宽
    unwrapped_img = np.zeros((unwrapped_height, unwrapped_width), dtype="u1")
    except_count = 0
    for j in range(unwrapped_width):
        theta = -2 * math.pi * (j / unwrapped_width)  # 1. start position such as "+ math.pi"
        # theta=theta+0.75*math.pi
        for i in range(90):
            unwrapped_radius = radius - i  # 2. don't forget
            x = unwrapped_radius * math.cos(theta) + x0  # 3. "sin" is clockwise but "cos" is anticlockwise
            y = unwrapped_radius * math.sin(theta) + y0
            x, y = int(x), int(y)
            try:
                unwrapped_img[i, j] = img[x, y]
            except Exception as e:
                except_count = except_count + 1
    return unwrapped_img[0:90,:]

#寻找字符区域
def FindCodeArea(img):
    h,w=img.shape
    result=cv2.bitwise_not(img)
    kernel=np.ones((h,830),dtype=np.float32)
    hist=signal.convolve2d(result,kernel,'valid')
    kernel01=np.ones((60,w),dtype=np.float32)
    hist01=signal.convolve2d(result,kernel01,'valid')
    hist01=hist01.ravel()
    maxIdx01=np.argmax(hist01)
    hist=hist.ravel()
    # plt.plot(hist)
    # plt.show()
    maxIdx=np.argmax(hist)
    result=result[10:60,maxIdx:maxIdx+835]
    return  result

#字符切割
def CutImg(img):
    h,w=img.shape
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # img = cv2.erode(img, kernel)
    #先将大的分割出来
    kernel=np.ones((h,15),dtype=np.float32)
    result=signal.convolve2d(img,kernel,'valid')
    result=cv2.threshold(result,1000,1,cv2.THRESH_BINARY)
    hist=result[1].ravel()
    # plt.plot(hist)
    # plt.show()
    startIdx=[]
    endIdx=[]
    for i in range(len(hist)):
        if(hist[i-1]==1)&(hist[i]==0):
            endIdx.append(i)
        if(hist[i-1]==0)&(hist[i]==1):
            startIdx.append(i)
    startIdx.insert(0,0)
    endIdx.append(w)
    imgarr=[]
    for i in range(len(startIdx)):
        tempimg=img[:,startIdx[i]:endIdx[i]]
        # cv2.imwrite(str(i)+'.jpg',tempimg)
        imgarr.append(tempimg)
        # cv2.waitKey()
    return imgarr

#图像的预处理和切割，输入的是灰度图
def FirstStep(img):
    result=ImagePorcessing(img)
    result=FindMaxArea(result)
    result=ExpandImg(result)
    result=FindCodeArea(result)
    imgarr=CutImg(result)
    return imgarr
#endregion


#region 识别字符
#再将已经分割好的照片再次切割为最终需要识别的照片，然后利用卷积神经网络和SVM进行识别
#卷积神经网络识别数据
import keras
from keras.utils import plot_model

# categories = ['0','1','2','3','4','5','6','7','8','9']
def printCategories(inputarr):
    categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # inputarr=inputarr.astype('int')
    inputarr=inputarr.ravel()
    idx=0
    for i in range(len(inputarr)):
        if(inputarr[i]>0.9)&(inputarr[i]<1.1):
            idx=i
            break
    return categories[idx]

model=keras.models.load_model('CNNModel.h5')

def ClassNum(img):
    h,w=img.shape
    array=[h,w]
    max=np.argmax([h,w])
    max=array[max]
    mask=np.zeros(shape=(max,max),dtype=np.uint8)
    x,y=(max-h)//2,(max-w)//2
    mask[0+x:h+x,0+y:w+y]=img[0:h,0:w]
    mask=cv2.resize(mask,dsize=(28,28))
    num=mask.reshape(-1,1,28,28)
    prediction=model.predict(num)
    return printCategories(prediction)

def CutNumArea(img):
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    img=cv2.erode(img,kernel)
    h, w = img.shape
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # img = cv2.erode(img, kernel)
    # 先将大的分割出来
    kernel = np.ones((h, 2), dtype=np.float32)
    result = signal.convolve2d(img, kernel, 'valid')
    result = cv2.threshold(result, 40, 1, cv2.THRESH_BINARY)
    hist = result[1].ravel()
    # plt.plot(hist)
    # plt.show()
    startIdx = []
    endIdx = []
    for i in range(1,len(hist)):
        if (hist[i - 1] == 1) & (hist[i] == 0):
            endIdx.append(i)
        if (hist[i - 1] == 0) & (hist[i] == 1):
            startIdx.append(i)
    # startIdx.insert(0, 0)
    endIdx.append(w)
    for i in range(len(startIdx)):
        if(endIdx[i]-startIdx[i]>50):
            mediu=(startIdx[i]+endIdx[i])//2
            startIdx.insert(i+1,mediu)
            endIdx.insert(i,mediu)
    imgarr = []
    for i in range(len(startIdx)):
        tempimg = img[:, startIdx[i]:endIdx[i]]
        imgarr.append(tempimg)
        # cv2.imshow(str(i),tempimg)
    # cv2.waitKey()
    return imgarr

def IdentityNum():
    img = cv2.imread('09_30_40.jpg', cv2.IMREAD_GRAYSCALE)
    imgarr = FirstStep(img)
    numsarr=[]
    for i in range(len(imgarr)):
        cv2.imshow(str(i),imgarr[i])
        if (i==2):continue
        if (i==2|3):
            continue
        tempimgarr=CutNumArea(imgarr[i])
        for j in range(len(tempimgarr)):
           numsarr.append(str(i)+':'+str(ClassNum(tempimgarr[j])))
    print(numsarr)
import time

import datetime
starttime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
print(starttime)
IdentityNum()
endtime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
print(endtime)
cv2.waitKey()
# print(endtime-starttime)
