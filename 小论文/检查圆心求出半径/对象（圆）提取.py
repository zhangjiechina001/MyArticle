import cv2
import numpy as np
#读取图片
src_img=cv2.imread('09_30_40.jpg',cv2.IMREAD_GRAYSCALE)#  circle.png  09_30_13.jpg
# src_img=cv2.pyrDown(src_img)
cv2.imshow('src',src_img)
#二值化
ret,binary=cv2.threshold(src_img,87,255,cv2.THRESH_BINARY)
#形态学操作
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
#CLOSE先膨胀再腐蚀
dst_img=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
#OPEN先腐蚀再膨胀
dst_img=cv2.morphologyEx(dst_img,cv2.MORPH_OPEN,kernel)
cv2.imshow('dst_img',dst_img)
#返回的是原图片，边界集合，轮廓的属性
_,contours,_=cv2.findContours(dst_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
result_img=np.zeros(src_img.shape,np.uint8)

for i in range(len(contours)):
    area=cv2.contourArea((contours[i]))
    # cv2.drawContours(result_img, contours, i, (255, 255, 255))
    if(area<3000):
        continue
    x,y,w,h=cv2.boundingRect(contours[i])
    ratio=0.0
    if(y!=0):
        ratio=float(w/h)

    if((ratio<1.1)&(ratio>0.9)):
        cv2.drawContours(result_img,contours,i,(255,255,255))
        print('面积:'+str(area))
        print('x:',x+w/2,'y:',y+h/2)
        arcLength=cv2.arcLength(contours[i],True)
        print('周长为：%f\n'%arcLength)
        reshapeValue=110
        imgSave=src_img[y-reshapeValue:y+h+reshapeValue,x-reshapeValue:x+w+reshapeValue]
        cv2.imshow('imgSave',imgSave)
        cv2.imwrite('imgSave.jpg',imgSave)
cv2.imshow('result_img',result_img)


cv2.waitKey()
cv2.destroyAllWindows()