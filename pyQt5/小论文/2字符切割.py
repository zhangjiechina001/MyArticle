import cv2
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
src=cv2.imread('afterCutImg.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('src',src)

#1.先将字符按照数量进行分类，进行不同的卷积核处理
h,w=src.shape
#反转二值化处理
bitwise=cv2.bitwise_not(src)
plt.imshow(bitwise)
cv2.imshow('bitwise',bitwise)
kernel=np.ones((h,15),dtype=np.float32)
result=signal.convolve2d(bitwise,kernel,'valid')
hist=result.ravel()
img1_4=bitwise[:,0:141]
img2_10=bitwise[:,152:470]
img3_1_L=bitwise[:,475:522]
img4_1_L=bitwise[:,530:576]
img5_6=bitwise[:,580:778]
img6_2=bitwise[:,783:w-1]
imgarr=[img1_4,img2_10,img3_1_L,img4_1_L,img5_6,img6_2]
for i in range(len(imgarr)):
    plt.subplot(2,3,i+1)
    img=cv2.cvtColor(imgarr[i],cv2.COLOR_GRAY2RGB)
    plt.imshow(img)
    cv2.imshow(str(i),img)
# plt.plot(hist)
plt.show()

#对划分好的每个子模块进行再次划分
#先写一个等分图片的程序
def CutImg(img,nums):
    h,w=img.shape
    leng=w//nums
    imgarr=[]
    for i in range(nums):
        temp=img[:,i*leng:(i+1)*leng]
        cv2.imshow('1_4_'+str(i),temp)
        imgarr.append(temp)
    return imgarr

#直方图分布切割图片
def histCutImg(img,imgName):
    h,w=img.shape
    kernel = np.ones((h, 1), dtype=np.float32)
    result = signal.convolve2d(img, kernel, 'valid')
    #对结果进行归一化处理,为0,1数列
    _,result=cv2.threshold(result,1,1,cv2.THRESH_BINARY)
    hist = result.ravel()
    startIdx=[]
    endIdx=[]
    for i in range(len(hist)-1):
        # if(i==len(hist)-1):
        #     continue
        #当前为0，下个为1，开始+1
        if(hist[i]==0)&(hist[i+1]==1):
            startIdx.append(i)
        # 当前为1，下个为0，结束+1
        if (hist[i] == 1) & (hist[i + 1] == 0):
            endIdx.append(i)
    for i in range(len(startIdx)):
        #如果发现字符粘连
        if(endIdx[i]-startIdx[i]>60):
            idx=int((startIdx[i]+endIdx[i])//2)
            startIdx.insert(i+1,idx)
            endIdx.insert(i,idx)
    print(startIdx)
    print(endIdx)
    for i in range(len(startIdx)):
        temp=img[:,startIdx[i]:endIdx[i]]
        cv2.imshow('temp'+str(i),temp)
        cv2.imwrite(imgName+str(i)+'.jpg',temp)
    plt.plot(hist)
    plt.show()

#对1照片进行分割
# CutImg(img5_6,6)
histCutImg(img2_10,'img2_10_')
cv2.waitKey()
cv2.destroyAllWindows()