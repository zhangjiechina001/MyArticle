import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
#%matplotlib inline

def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256]) #image.ravel()#ravel函数功能是将多维数组降为一维数组,统计各个bin的频次，256：bin的个数，[0, 256]：范围
    plt.show("直方图") #和OpenCV中的想要的直方图不同
"""
画灰度图直方图:

绘图都可以调用matplotlib.pyplot库来进行，其中的hist函数可以直接绘制直方图。

plt.hist(arr, bins=50, normed=1, facecolor='green', alpha=0.75)

hist的参数非常多，但常用的就这五个，只有第一个是必须的，后面四个可选

arr: 需要计算直方图的一维数组

bins: 直方图的柱数，可选项，默认为10

normed: 是否将得到的直方图向量归一化。默认为0

range参数表示箱子的下限和上限。即横坐标显示的范围，范围之外的将被舍弃
"""

def image_hist(image):
    color = ('blue', 'green', 'red')  #图像三通道
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256]) #绘制各个通道的直方图
        plt.plot(hist, color=color) #定义线的颜色
        plt.xlim([0, 256]) #x轴的范围
    plt.show()
"""
calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) 
images参数表示输入图像，传入时应该用中括号[ ]括起来

channels参数表示传入图像的通道，如果是灰度图像，那就不用说了，只有一个通道，值为0，
如果是彩色图像（有3个通道），那么值为0,1,2,中选择一个，对应着BGR各个通道。这个值也得用[ ]传入。

mask参数表示掩膜图像。如果统计整幅图，那么为None。
主要是如果要统计部分图的直方图，就得构造相应的掩膜来计算。

histSize参数表示灰度级的个数，需要中括号，比如[256]

ranges参数表示像素值的范围，通常[0,256]。此外，假如channels为[0,1],ranges为[0,256,0,180],
则代表0通道范围是0-256,1通道范围0-180。

hist参数表示计算出来的直方图。

"""


src = cv.imread("F:/images/lena.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
plot_demo(src)
image_hist(src)
cv.waitKey(0)

cv.destroyAllWindows()

