import cv2


# 寻找最大轮廓，传入的是一个二值的黑白图
def FindBigestContour(src):
    imax = 0
    imaxcontours = -1
    # 返回的是原图片，边界集合，轮廓的属性
    image, contours, hierarchy = cv2.findContours(src,
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        itemp = cv2.contourArea(contours[i])
        if (imaxcontours < itemp):
            imaxcontours = itemp
            imax = i

    return contours[imax]

src=cv2.imread('bestimg.jpg')
src=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
_,src=cv2.threshold(src,150,255,cv2.THRESH_BINARY)
# cv2.imshow('temp',src)
maxContour=FindBigestContour(src)
_,contours,_=cv2.findContours(src,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
import numpy as np
white = np.zeros(src.shape, np.float32)
for i in range(len(contours)):
    (x,y,w,h)=cv2.boundingRect(contours[i])
    cv2.rectangle(white,(x,y),(x+w,y+h),color=(255,0,244))

# src=cv2.drawContours(white,contours[1],-1,color=(100,190,200),thickness=3)
cv2.imshow('result',white)
cv2.waitKey()
cv2.destroyAllWindows()