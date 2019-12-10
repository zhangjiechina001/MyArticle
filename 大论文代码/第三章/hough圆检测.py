import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('09_30_13.jpg',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# 第一个参数image是输入图像矩阵，要求是灰度图像；
#
# 第二个参数 circles是一个包含检测到的圆的信息的向量，向量内第一个元素是圆的横坐标，第二个是纵坐标，第三个是半径大小；
#
# 第三个参数 methodmethod是所使用的圆检测算法，目前只有CV_HOUGH_GRADIENT一个可选；
#
# 第四个参数 dp是累加面与原始图像相比的分辨率的反比参数，dp=2时累计面分辨率是元素图像的一半，宽高都缩减为原来的一半，dp=1时，两者相同。（关于这个分辨率的概念没有理解透，按道理低分辨率应该意味着更快的检测速度，然而实测恰恰相反）
#
# 第五个参数 minDist定义了两个圆心之间的最小距离；
#
# 第六个参数param1是Canny边缘检测的高阈值，低阈值被自动置为高阈值的一半；
#
# 第七个参数param2是累加平面对是否是圆的判定阈值；
#
# 第八和第九个参数定义了检测到的圆的半径的最大值和最小值；
#(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=100,minRadius=700,maxRadius=840)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,255),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,255,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()