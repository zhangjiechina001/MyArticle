import numpy as np
import math
import cv2
def unFoldImg(img):
    # 得到圆形区域的中心坐标
    x0 = img.shape[0] // 2
    y0 = img.shape[1] // 2
    unwrapped_height = radius = img.shape[0] // 2  # 展开后的高
    unwrapped_width = int(2 * math.pi * radius)  # 展开后的宽
    unwrapped_img = np.zeros((unwrapped_height, unwrapped_width, 3), dtype="u1")
    except_count = 0
    for j in range(unwrapped_width):
        theta = -2 * math.pi * (j / unwrapped_width)  # 1. 开始位置
        # theta=theta+0.75*math.pi
        for i in range(unwrapped_height -425//2):
            unwrapped_radius = radius - i  # 2. don't forget
            x = unwrapped_radius * math.cos(theta) + x0  # 3. "sin" is clockwise but "cos" is anticlockwise
            y = unwrapped_radius * math.sin(theta) + y0
            x, y = int(x), int(y)
            try:
                unwrapped_img[i, j, :] = img[x, y, :]
            except Exception as e:
                except_count = except_count + 1
    print(except_count)
    return unwrapped_img

src=cv2.imread('imgSave.jpg')

import datetime
#添加处理时间信息
def dealTime(fun,**kwargs):
    start=datetime.datetime.now()
    imgData=fun(**kwargs)
    end=datetime.datetime.now()
    totalMs=(end-start).total_seconds()
    ret=[imgData,totalMs*1000]
    return ret

src=cv2.pyrDown(src)
src=cv2.pyrDown(src)
unwrappedImg=dealTime(unFoldImg,img=src)
cv2.namedWindow('unFloadImg',cv2.WINDOW_NORMAL)
cv2.imshow('unFloadImg',unwrappedImg[0])
print('时间{0}ms'.format(str(unwrappedImg[1])))
cv2.waitKey()
cv2.destroyAllWindows()