import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
#作图
def unfloodImgPlt(imgInfoList):
    plt.figure()
    i=0
    for imginfo in imgInfoList:
        blurName,imgData,dealTime=imginfo
        # imgData = cv2.cvtColor(imgData, cv2.COLOR_GRAY2BGR)
        i=i+1
        plt.subplot(2,1,i)
        plt.imshow(imgData)
        plt.xticks([])
        plt.yticks([])
        tempyime="%0.3f" % dealTime
        plt.title("{0}".format(blurName), fontsize=20)
        # cv2.imshow(blurName,imgData)
    plt.show()

import datetime
# 添加处理时间信息
def dealTime(fun, **kwargs):
    start = datetime.datetime.now()
    imgData = fun(**kwargs)
    end = datetime.datetime.now()
    totalMs = (end - start).total_seconds()
    ret = [imgData, totalMs * 1000]
    return ret

def unFlodImgPro(img,startTheta,endTheta):# 得到圆形区域的中心坐标
    x0 = img.shape[0] // 2
    y0 = img.shape[1] // 2
    unwrapped_height = radius = img.shape[0] // 2  # 展开后的高
    full_width= int(2 * math.pi * radius)   #总长
    unwrapped_width = int(2 * math.pi * radius*(endTheta-startTheta)/(2*math.pi))  # 展开后的宽
    unwrapped_img = np.zeros((unwrapped_height-750, unwrapped_width, 3), dtype="u1")
    except_count = 0
    for j in range(unwrapped_width):
        theta = -2 * math.pi * (j / full_width)+startTheta  # 1. 开始位置
        # theta=theta+0.75*math.pi
        for i in range(unwrapped_height-750):
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
startTheta=0.8*math.pi
endTheta=startTheta+2*math.pi
retimg=dealTime(unFlodImgPro,img=src,startTheta=startTheta,endTheta=endTheta)
startTheta=1*math.pi
endTheta=startTheta+2*math.pi
retimg1=dealTime(unFlodImgPro,img=src,startTheta=startTheta,endTheta=endTheta)
cv2.namedWindow('unFloadImgPro',cv2.WINDOW_NORMAL)
cv2.imshow('unFloadImgPro',retimg[0])
nameList=['异常切割','正常切割']
imgInfoList=[retimg,retimg1]
fullImgInfoList=[]
#将信息添加到数组里面
for i in range(len(nameList)):
    imgData,time=imgInfoList[i]
    name=nameList[i]
    imgFullInfo=[name,imgData,time]
    fullImgInfoList.append(imgFullInfo)

print('时间{0}ms'.format(str(retimg[1])))
unfloodImgPlt(fullImgInfoList)

cv2.waitKey()
cv2.destroyAllWindows()