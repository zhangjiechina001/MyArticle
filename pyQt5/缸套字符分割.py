import cv2
import numpy as np
import matplotlib.pyplot as plt

# 对图片进行垂直分割
def verticalCut(img,img_num):
    (x,y)=img.shape #返回的分别是矩阵的行数和列数，x是行数，y是列数
    pointCount=np.zeros(y,dtype=np.float32)#每列黑色的个数
    x_axes=np.arange(0,y)
    #i是列数，j是行数
    tempimg=img.copy()
    for i in range(0,y):
        for j in range(0,x):
            # if j<15:
            if(tempimg[j,i]<=10):
                pointCount[i]=pointCount[i]+1
    figure=plt.figure(str(img_num))
    for num in range(pointCount.size):
        pointCount[num]=pointCount[num]
        if(pointCount[num]<0):
            pointCount[num]=0
    plt.plot(x_axes,pointCount)
    start = []
    end = []
    # 对照片进行分割
    print(pointCount)
    for index in range(1, y-1):
        # 上个为0当前不为0，即为开始
        if ((pointCount[index-1] == 0) & (pointCount[index] != 0)):
            start.append(index)
        # 上个不为0当前为0，即为结束
        elif ((pointCount[index] != 0) & (pointCount[index +1] == 0)):
            end.append(index)
    imgArr=[]
    # for idx in range(0,len(start)):
    #     tempimg=img[ :,start[idx]:end[idx]]
    #     cv2.imshow(str(img_num)+"_"+str(idx), tempimg)
    #     # cv2.imwrite(img_num+'_'+str(idx)+'.jpg',tempimg)
    #     imgArr.append(tempimg)
    plt.show()
    return imgArr

def callBack(x):
    global src
    tempsrc=src.copy()
    tempsrcaa = cv2.morphologyEx(src, cv2.MORPH_OPEN, (x, x))
    cv2.imshow('src',tempsrcaa)

def drawContours(binarysrc):
    _,contours,_=cv2.findContours(binarysrc,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    tempimg=np.zeros(binarysrc.shape,np.float32)
    # area = cv2.contourArea((contours[i]))
    for i in range(len(contours)):
        if(cv2.contourArea(contours[i])<0):
            continue
        x,y,w,h=cv2.boundingRect(contours[i])
        cv2.rectangle(tempimg,(x,y),(x+w,y+h),color=(255,0,190))
    cv2.imshow('dealedImg',tempimg)

src=cv2.imread('bestimg.jpg',cv2.IMREAD_GRAYSCALE)
src=src[12:40,1100:1550]

cv2.namedWindow('src',cv2.WINDOW_NORMAL)
# cv2.createTrackbar('resize','src',2,10,callBack)
kernel=np.ones((2,2),np.uint8)
# src=cv2.morphologyEx(src,cv2.MORPH_OPEN,kernel)
src=cv2.dilate(src,kernel)
drawContours(src)
# canny=cv2.Canny(src,10,30)
# cv2.imshow('canny',canny)
cv2.imshow('src',src)
verticalCut(src,0)
cv2.waitKey()
cv2.destroyAllWindows()
