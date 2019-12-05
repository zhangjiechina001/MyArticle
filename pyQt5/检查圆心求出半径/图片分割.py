import cv2
import matplotlib.pyplot as plt

src=cv2.imread('imgSave.jpg',cv2.IMREAD_GRAYSCALE)
w,h=src.shape
src=src[0:w//4,h//2:h-1]
src=cv2.cvtColor(src,cv2.COLOR_GRAY2RGB)
_,binary=cv2.threshold(src,142,255,cv2.THRESH_BINARY)
#二值图
plt.subplot(221)
plt.imshow(src)
plt.xlabel('(a)原图',fontproperties='SimHei',fontsize=15)
plt.xticks([])
plt.yticks([])
plt.subplot(222)
plt.imshow(binary)
plt.xlabel('(b)二值图',fontproperties='SimHei',fontsize=15)
plt.xticks([])
plt.yticks([])
plt.subplot(223)
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
closeOperation=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
plt.imshow(closeOperation)
plt.xlabel('(c)去噪处理',fontproperties='SimHei',fontsize=15)
plt.xticks([])
plt.yticks([])
plt.subplot(224)
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
openOperation=cv2.morphologyEx(closeOperation,cv2.MORPH_OPEN,kernel)
plt.imshow(openOperation)
plt.xlabel('(d)目标增强',fontproperties='SimHei',fontsize=15)
plt.xticks([])
plt.yticks([])
plt.show()
