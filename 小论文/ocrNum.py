import cv2
import numpy as np
import matplotlib.pyplot as plt

#图像二值化处理
def imgThreshold(img):
    rosource,binary=cv2.threshold(img,121,255,cv2.THRESH_BINARY)
    return binary

#1.先水平分割，再垂直分割
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
            if(tempimg[j,i]==0):
                pointCount[i]=pointCount[i]+1
    figure=plt.figure(str(img_num))
    # for num in range(pointCount.size):
    #     pointCount[num]=pointCount[num]
    #     if(pointCount[num]<0):
    #         pointCount[num]=0
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
    for idx in range(0,len(start)):
        tempimg=img[ :,start[idx]:end[idx]]
        cv2.imshow(str(img_num)+"_"+str(idx), tempimg)
        cv2.imwrite(img_num+'_'+str(idx)+'.jpg',tempimg)
        imgArr.append(tempimg)
    return imgArr
    # cv2.waitKey()
    # plt.show()

#对图片进行水平分割,返回的是照片数组
def horizontalCut(img):
    (x,y)=img.shape #返回的分别是矩阵的行数和列数，x是行数，y是列数
    pointCount=np.zeros(y,dtype=np.uint8)#每行黑色的个数
    x_axes=np.arange(0,y)
    for i in range(0,x):
        for j in range(0,y):
            if(img[i,j]==0):
                pointCount[i]=pointCount[i]+1
    plt.plot(pointCount,x_axes)
    start=[]
    end=[]
    #对照片进行分割
    print(pointCount)
    for index in range(1,y):
        #上个为0当前不为0，即为开始
        if((pointCount[index]!=0)&(pointCount[index-1]==0)):
            start.append(index)
        #上个不为0当前为0，即为结束
        elif((pointCount[index]==0)&(pointCount[index-1]!=0)):
             end.append(index)
    # img1=img[start[0]:end[0],:]
    # img2=img[start[1]:end[1],:]
    # img3=img[start[2]:end[2],:]
    imgArr=[]
    for m in range(len(start)):
        tempimg=img[start[m]:end[m],:]
        imgArr.append(tempimg)
        cv2.imshow(str(m),imgArr[m])
    cv2.waitKey()
    plt.show()
    return imgArr

#输入的分别是原图模板和标签
def matchTemplate(src,matchSrc,label):
    binaryc=imgThreshold(src)
    result=cv2.matchTemplate(binaryc,matchSrc,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    tw,th=matchSrc.shape[:2]
    tl=(max_loc[0]+th+2,max_loc[1]+tw+2)
    cv2.rectangle(src,max_loc,tl,[0,0,0])
    cv2.putText(src,label,max_loc,fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.6,
                color=(240,230,0))
    cv2.imshow('001',src)

img = cv2.imread("bestimg.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
binary=imgThreshold(img)
horizontalCut(binary)
cv2.imshow('result',binary)
match=cv2.imread('num_1_1.jpg',cv2.COLOR_BGR2GRAY)
matchTemplate(img,match,'5')

cv2.waitKey()
cv2.destroyAllWindows()






