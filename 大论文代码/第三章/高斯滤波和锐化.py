import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
def fourBlur(imgList):
    BlurName=['原图','高斯滤波','中值滤波','均值滤波']
    plt.figure()
    for i in range(len(BlurName)):
        plt.subplot(2,2,i+1)
        plt.imshow(imgList[i])
        plt.xticks([]);
        plt.yticks([])
        plt.title(BlurName[i],fontsize=25)
    plt.show()

def sharpImg(imgList):
    BlurName = ['原图', '高斯滤波后锐化', '中值滤波后锐化', '均值滤波后锐化']
    plt.figure()
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    for i in range(len(BlurName)):
        dst=cv2.filter2D(imgList[i],-1,kernel=kernel)
        if(i==0):
            dst=img_list[i]
        plt.subplot(2, 2, i + 1)
        plt.imshow(dst)
        plt.xticks([]);
        plt.yticks([])
        plt.title(BlurName[i],fontsize=25)
    plt.show()





img_cpurce=cv2.imread('gasuss_noise.jpg')
img_Gauss=cv2.GaussianBlur(img_cpurce,(7,7),0)
img_Media=cv2.medianBlur(img_cpurce,7)
img_Average=cv2.blur(img_cpurce,(5,5))
img_list=[img_cpurce,img_Gauss,img_Media,img_Average]
fourBlur(imgList=img_list)
sharpImg(imgList=img_list)