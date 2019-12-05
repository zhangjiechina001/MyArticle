import numpy as np
import math
import cv2

#回调事件
def thresholdeEvent(x):
    global src
    srcimg=src
    src=cv2.cvtColor(srcimg,cv2.COLOR_BGR2GRAY)
    reet,binary=cv2.threshold(src,x,255,cv2.THRESH_BINARY)
    cv2.imshow('binary', binary)

#将图像转化为正方形
def resizeSquare(srcimg):
    src=cv2.cvtColor(srcimg,cv2.COLOR_BGR2GRAY)
    reet,binary=cv2.threshold(src,110,255,cv2.THRESH_BINARY)
    cv2.namedWindow('binary')
    cv2.createTrackbar('binary','threshold',100,255,thresholdeEvent)
    cv2.imshow('binary',binary)


img = cv2.imread("unFloodAfterBinary.jpg")
# img=cv2.pyrDown(img)
# img=cv2.pyrDown(img)
resizeSquare(img)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("panoramagram", img)
# 得到圆形区域的中心坐标
x0 = img.shape[0] // 2
y0 = img.shape[1] // 2
# 通过圆形区域半径构造展开后的图像
unwrapped_height = radius = img.shape[0] // 2#展开后的高
unwrapped_width = int(2 * math.pi * radius)#展开后的宽
unwrapped_img = np.zeros((unwrapped_height, unwrapped_width, 3), dtype="u1")
except_count = 0
for j in range(unwrapped_width):
    theta = -2 * math.pi * (j / unwrapped_width)  # 1. start position such as "+ math.pi"
    # theta=theta+0.75*math.pi
    for i in range(unwrapped_height-850):
        unwrapped_radius = radius - i  # 2. don't forget
        x = unwrapped_radius * math.cos(theta) + x0  # 3. "sin" is clockwise but "cos" is anticlockwise
        y = unwrapped_radius * math.sin(theta) + y0
        x, y = int(x), int(y)
        try:
            unwrapped_img[i, j, :] = img[x, y, :]
        except Exception as e:
            except_count = except_count + 1
print(except_count)
cv2.namedWindow('Unwrapped',cv2.WINDOW_NORMAL)
cv2.imshow("Unwrapped", unwrapped_img)
cv2.imwrite('unFlood.jpg',unwrapped_img)
cv2.waitKey(0)

