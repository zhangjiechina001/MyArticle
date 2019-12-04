import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  #
def drawHist(img):
    hsit=cv2.calcHist([img],[0],None,[256],[0,255])
    plt.plot(hsit,color='r')
    plt.show()
    plt.title('直方图',fontsize=20)

# src=cv2.imread('moution.png')
# histb=cv2.calcHist([src],[0],None,[256],[0,255])
# histg=cv2.calcHist([src],[1],None,[256],[0,255])
# histr=cv2.calcHist([src],[2],None,[256],[0,255])
#
# plt.plot(histb,color='b')
# plt.plot(histg,color='g')
# plt.plot(histr,color='r')
# plt.show()
src=cv2.imread('09_30_13.jpg')
drawHist(src)