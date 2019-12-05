import cv2
import numpy as np
import matplotlib.pyplot as plot
from scipy import signal



src=cv2.imread('img_0.jpg',cv2.IMREAD_GRAYSCALE)
src=cv2.bitwise_not(src)
srctemp=src.copy()
kernel01=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
src=cv2.erode(src,kernel01)
# src=src[10:40,:]
h,w=src.shape
#定义卷积核
kernel=np.ones((h,1),dtype=np.float32)
afterDeal=signal.convolve2d(src,kernel,'valid')//256
plot.plot(afterDeal.ravel())
end01=afterDeal.ravel()
#对图片进行切割
def imgCut(img):
    startIdx=[]
    endIdx=[]
    zero=0
    for i in range(len(end01)-1):
        # if (end01[i - 1]<1):
        #     startIdx.append(i)
        if((end01[i]<1)&(end01[i+1]>1)):
            startIdx.append(i)
        elif((end01[i]==0)&(end01[i-1]!=0)):
            endIdx.append(i)
    for j in range(len(endIdx)):
        img=srctemp[:,startIdx[j]:endIdx[j]+5]
        cv2.imshow('img_'+str(j),img)
        path=r'cimg_0_'+str(j)+'.jpg'
        cv2.imwrite(path,img)
    print(startIdx,endIdx)
imgCut(src)
plot.show()