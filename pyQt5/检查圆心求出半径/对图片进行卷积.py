import cv2
import numpy as np
import matplotlib.pyplot as plot
from scipy import signal
src=cv2.imread('bestimg.jpg',cv2.IMREAD_GRAYSCALE)

# _,src=cv2.threshold(src,150,1,type=cv2.THRESH_BINARY)
plot.subplot(311)
plot.imshow(src)
tempsrc=src[30:60,:]
tempsrc=cv2.bitwise_not(tempsrc)
cv2.imshow('imgCuted',tempsrc)
kernel=np.ones((30,1),dtype=np.float32)
# dst=cv2.filter2D(tempsrc,cv2.CV_8U,kernel=kernel,borderType=cv2.BORDER_DEFAULT)
end=signal.convolve2d(tempsrc,kernel,'valid')
end=end/256
kernel01=np.ones(18,dtype=np.uint8)
end01=np.convolve(end.ravel(),kernel01,'full')
plot.subplot(313)
plot.xlim(0,1100)
plot.plot(end01)
# for i in range(0,1100):
#     print(end01[i])

plot.subplot(312)
plot.xlim(0,1100)
plot.plot(end.ravel())
def imgCut(img):
    startIdx=[]
    endIdx=[]
    zero=0
    for i in range(len(end01)-100):
        # if (end01[i - 1]<1):
        #     startIdx.append(i)
        if((end01[i]<1)&(end01[i+1]>=1)):
            startIdx.append(i)
        if((end01[i]<=1)&(end01[i-1]>=1)):
            endIdx.append(i)
    for j in range(len(endIdx)):
        img=src[:,startIdx[j]:endIdx[j]]
        cv2.imshow('img_'+str(j),img)
        path=r'cuted\img_'+str(j)+'.jpg'
        cv2.imwrite(path,img)

    print(startIdx,endIdx)
src=src[10:,:]
imgCut(src)
plot.show()

cv2.waitKey()
cv2.destroyAllWindows()