import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
src=cv2.imread('cut1.jpg')
#去燥
# kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))
# src=cv2.dilate(src,kernel)
h,w,channels=src.shape
cv2.namedWindow('src',cv2.WINDOW_NORMAL)
cv2.imshow('src',src)
src=cv2.bitwise_not(src)#将图片颜色转换
#d对原图片进行卷积处理,h:60
kernel=np.ones((h,850),dtype=np.float32)
tempsrc=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
result=signal.convolve2d(tempsrc,kernel,'valid')
# result=result/100
plt.plot(result.ravel())
hist=result.ravel()
maxIdx=np.argmax(hist)
print(maxIdx)
aftercut=src[:,maxIdx:860+maxIdx,:]
cv2.imshow('afterCut',aftercut)
# aftercut=cv2.cvtColor(aftercut,cv2.COLOR_GRAY2BGR)
cv2.imwrite('afterCutImg.png',aftercut)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()