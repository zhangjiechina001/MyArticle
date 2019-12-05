import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  #
fontsize=15
def drawHist(img):
    plt.subplot(3,1, 1)
    hsit=cv2.calcHist([img],[0],None,[256],[0,255])
    plt.plot(hsit,color='r')
    plt.title('零件整体灰度直方图',fontsize=fontsize)

def drawCodeHist(img):
    plt.subplot(3,1, 2)
    hsit = cv2.calcHist([img], [0], None,[256], [0, 255])
    plt.plot(hsit,color='r')
    plt.title('字符细节灰度直方图',fontsize=fontsize)

def drawCodeMaskHist(img,mask):
    plt.subplot(3,1, 3)
    hsit = cv2.calcHist([img], [0], mask,[256], [0, 255])
    plt.plot(hsit,color='r')
    plt.title('掩膜处理后字符细节灰度直方图',fontsize=fontsize)

src=cv2.imread('09_30_13.jpg')
CodeSrc=cv2.imread('detail.jpg',cv2.IMREAD_GRAYSCALE)
thresh,mask_img=cv2.threshold(src=CodeSrc,thresh=73,maxval=255,type=cv2.THRESH_BINARY)

drawHist(src)
drawCodeHist(CodeSrc)
drawCodeMaskHist(CodeSrc,mask_img)
plt.show()