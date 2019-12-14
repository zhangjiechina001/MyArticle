import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
#两次缩小
def reduceImg(img):
    ret_img=cv2.pyrDown(img)
    ret_img=cv2.pyrDown(ret_img)
    return ret_img
#得到二值图
def binaryImage(img,binary_type):
    ret,binary=cv2.threshold(img,141,255,binary_type)
    return ret,binary

# 提取圆心，半径
def getPointAndR(src):
    _, contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result_img = np.zeros(src.shape,np.uint8)
    retImg=None
    i=1
    circle_point = None
    circle_r = None
    # for contour in contours:
    #     area=cv2.contourArea(contour)
    #     print('area{0}:{1}'.format(i,str(area)))
    #     i+=1
    resizeNum=16

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        print(area)
        if (area < 1500000//resizeNum or area>2500000//resizeNum):
            continue
        x, y, w, h = cv2.boundingRect(contours[i])
        ratio = 0.0
        if (y != 0):
            ratio = float(w / h)
        if ((ratio > 0.95) & (ratio < 1.05)):
            cv2.drawContours(result_img, contours, i, (255,255, 255),thickness=5)
            # cv2.namedWindow('cut',cv2.WINDOW_NORMAL)
            # cv2.imshow('cut',result_img)
            # if(area>2000000 and area<2500000):
            size = 100
            # else:
            #     size=0
            # retImg = readImg[y - size:y + h + size, x - size:x + w + size]
            circle_point=(y+w//2,x+h//2)
            circle_r=w//2
            break
    return circle_point,circle_r,result_img

#输入图片，圆心点，展开内半径，展开宽度
def unfloodImage(img,point,unfloodR,width,startTheta):
    h,w=img.shape
    x0,y0=point
    unwrapped_width=unfloodR+width#展开的最大半径
    unwrapped_height=width
    full_width=int(2*math.pi*unwrapped_width)#展开后的长度
    unwrapped_img=np.zeros((unwrapped_height,full_width),dtype='u1')
    except_count=0
    for j in range(full_width):
        theta = -2 * math.pi * (j / full_width)-startTheta  # 1. 开始位置
        # theta=theta+0.75*math.pi
        for i in range(unwrapped_height):
            unwrapped_radius = unwrapped_width -i  # 2. don't forget
            x = unwrapped_radius * math.cos(theta) + x0  #
            y = unwrapped_radius * math.sin(theta) + y0
            x, y = int(x), int(y)
            try:
                if x<0 or x>=h or y<0 or y>=w:
                    continue
                unwrapped_img[i, j] = img[x, y]
            except Exception as e:
                except_count = except_count + 1
    print('expect count:'+str(except_count))
    return unwrapped_img

#返回计算极值，开始角度，卷积结果
def calcMax(img):
    h,w,_=img.shape
    kernel=np.ones((h,220),dtype=np.float32)
    # kernel[0:10,:]=0
    result = signal.convolve2d(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), kernel, 'valid')
    hist=result.ravel()
    ret_max=np.max(hist)
    start_position=np.argmax(hist)
    # start_position=start_position[0]
    ret_theta=(start_position/w)*2*np.pi
    return ret_max,ret_theta,hist

def last_fun(img):
    copyImg=img.copy()
    #两次缩小
    smallImg=reduceImg(img)
    plt.subplot(2,4,1)
    plt.imshow(smallImg,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("两次缩小图", fontsize=10)
    #二值化
    thresh,binary=binaryImage(smallImg,binary_type=cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    plt.subplot(2, 4, 2)
    plt.imshow(binary, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("二值化阈值：{0}".format(str(thresh)), fontsize=10)
    #提取圆心，半径
    circle_point, circle_r,drawImg=getPointAndR(binary)
    plt.subplot(2, 4, 3)
    plt.imshow(drawImg,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("圆心:{0},半径：{1}".format(str(circle_point),str(circle_r)), fontsize=10)
    #分角度展开灰度图
    #1.展开角为0
    plt.figure()
    unflood_img1=unfloodImage(smallImg,point=circle_point,unfloodR=circle_r,width=40,startTheta=0)
    unflood_img1 = cv2.Canny(unflood_img1,100,200)
    thresh=0
    plt.subplot(4,1, 1)
    plt.imshow(unflood_img1,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("第二次展开,thresh:{0}".format(str(thresh)), fontsize=10)

    #2.展开角为90°
    unflood_img2=unfloodImage(smallImg,point=circle_point,unfloodR=circle_r,width=40,startTheta=math.pi/2)
    unflood_img2=cv2.Canny(unflood_img2,100,200)
    plt.subplot(4,1, 2)
    plt.imshow(unflood_img2,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("第二次展开,thresh:{0}".format(str(thresh)), fontsize=10)
    plt.show()


if __name__=='__main__':
    img=cv2.imread('OKPictures//16_55_45.jpg',cv2.IMREAD_GRAYSCALE)
    last_fun(img)