import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal

#对原图片进行卷积，得出定位点
dst_img=cv2.imread('afterCutImg.png',cv2.IMREAD_GRAYSCALE)
h,w=dst_img.shape
# src=cv2.bitwise_not(dst_img)
src=dst_img
cv2.namedWindow('src',cv2.WINDOW_NORMAL)
cv2.imshow('src',dst_img)
kernel=np.ones((h,5),dtype=np.float32)
result=signal.convolve2d(src,kernel,'valid')

plt.subplot2grid((3,1),(0,0))
plt.imshow(cv2.cvtColor(dst_img,cv2.COLOR_GRAY2RGB))
plt.xlabel('(a)原图',fontproperties='SimHei',fontsize=25)
plt.xticks([])
plt.yticks([])


plt.subplot2grid((3,1),(1,0))
hist=result.ravel()
plt.plot(hist)
plt.xlabel('(b)卷积投影分布',fontproperties='SimHei',fontsize=25)
# plt.xlim((0,w))

plt.subplot2grid((3,1),(2,0))
result=cv2.threshold(result,10000,1,cv2.THRESH_BINARY)
hist=result[1].ravel()
plt.plot(hist)
plt.xlabel('(c)二值化分布',fontproperties='SimHei',fontsize=25)
# plt.xticks([])
# plt.yticks([])





# cv2.imwrite('afterCutImg.png',cutimg)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()