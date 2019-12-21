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

def unfloodImagePro(img,point,unfloodR,width,startTheta):
    h, w = img.shape
    x0, y0 = point
    unwrapped_width = unfloodR + width  # 展开的最大半径
    unwrapped_height = width
    full_width = int(2 * math.pi * unwrapped_width)  # 展开后的长度
    unflood_width=950
    unwrapped_img = np.zeros((unwrapped_height, unflood_width), dtype='u1')
    except_count = 0
    for j in range(unflood_width):
        theta = -2 * math.pi * (j / full_width) - startTheta  # 1. 开始位置
        # theta=theta+0.75*math.pi
        for i in range(unwrapped_height):
            unwrapped_radius = unwrapped_width - i  # 2. don't forget
            x = unwrapped_radius * math.cos(theta) + x0  #
            y = unwrapped_radius * math.sin(theta) + y0
            x, y = int(x), int(y)
            try:
                if x < 0 or x >= h or y < 0 or y >= w:
                    continue
                unwrapped_img[i, j] = img[x, y]
            except Exception as e:
                except_count = except_count + 1
    print('expect count:' + str(except_count))
    return unwrapped_img

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
    h,w=img.shape
    kernel=np.ones((h,230),dtype=np.float32)
    # kernel[0:10,:]=0
    result = signal.convolve2d(img, kernel, 'valid')
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
    plt.subplot(1,3,1)
    plt.imshow(smallImg,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("两次缩小图", fontsize=10)
    #二值化
    thresh,binary=binaryImage(smallImg,binary_type=cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    plt.subplot(1, 3, 2)
    plt.imshow(binary, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("二值化阈值：{0}".format(str(thresh)), fontsize=10)
    #提取圆心，半径
    circle_point, circle_r,drawImg=getPointAndR(binary)
    plt.subplot(1, 3, 3)
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
    plt.subplot(4,1,1)
    plt.imshow(unflood_img1,cmap='gray')
    # plt.xticks([])
    # plt.yticks([])
    plt.title("第一次展开,thresh:{0}".format(str(thresh)), fontsize=10)
    #绘制直方分布图
    plt.subplot(4, 1, 2)
    ret_max1, ret_theta1, hist1=calcMax(unflood_img1)
    plt.plot(hist1)
    plt.title('max:{0},theta:{1}'.format(str(ret_max1),str(ret_theta1)),fontsize=10)



    #2.展开角为90°
    unflood_img2=unfloodImage(smallImg,point=circle_point,unfloodR=circle_r,width=40,startTheta=math.pi/2)
    unflood_img2=cv2.Canny(unflood_img2,100,200)
    plt.subplot(4,1,3)
    plt.imshow(unflood_img2,cmap='gray')
    # plt.xticks([])
    # plt.yticks([])
    plt.title("第二次展开,thresh:{0}".format(str(thresh)), fontsize=10)

    # 绘制直方分布图
    plt.subplot(4, 1, 4)
    ret_max2, ret_theta2, hist2 = calcMax(unflood_img2)
    plt.plot(hist2)
    plt.title('max:{0},theta:{1}'.format(str(ret_max2), str(ret_theta2+math.pi/2)), fontsize=10)

    _font_size=20
    #确定展开角度
    plt.figure()
    plt.subplot(4,1,1)
    # theta=0.0
    if(ret_max1>ret_max2):
        theta=ret_theta1
    else:
        theta=ret_theta2+math.pi/2
    #确定大图展开位置
    # circle_point, circle_r, drawImg
    point=(circle_point[0]*4,circle_point[1]*4)
    r=circle_r*4

    #看圆的半径，得出型号，一个为109，一个为208,208的字符高度为50左右，109的为80左右
    unflood_srcImg=unfloodImagePro(copyImg,point=point,unfloodR=r,width=130,startTheta=theta-0.03)
    if(circle_r>204):
        unflood_srcImg=unflood_srcImg[50:110,:]
    else:
        unflood_srcImg=unflood_srcImg[0:100,:]

    plt.imshow(unflood_srcImg,cmap='gray')
    plt.title('原图展开{0}'.format(str(theta)),fontsize=_font_size)

    plt.subplot(4, 1, 2)
    binaryUnfloodImg=cv2.adaptiveThreshold(unflood_srcImg,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,blockSize=17,C=10)
    # cv2.imshow('img',tempimg)
    # binaryUnfloodImg=binaryImage(unflood_srcImg,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    plt.imshow(binaryUnfloodImg,cmap='gray')
    plt.title('局部自适应二值化',fontsize=_font_size)

    plt.subplot(4, 1, 3)
    #形态学处理
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dst_img=binaryUnfloodImg
    # CLOSE先膨胀再腐蚀
    dst_img = cv2.morphologyEx(dst_img, cv2.MORPH_CLOSE, kernel)
    # OPEN先腐蚀再膨胀
    dst_img = cv2.morphologyEx(dst_img, cv2.MORPH_OPEN, kernel)
    # dst_img=cv2.Canny(dst_img,100,200)
    plt.imshow(dst_img, cmap='gray')
    plt.title('形态学开操作:kernel={0}'.format(str((2,2))), fontsize=_font_size)

    plt.subplot(4, 1, 4)
    #字符切割
    _, contours, _ = cv2.findContours(dst_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count=0
    unflood_srcImg=cv2.cvtColor(unflood_srcImg,cv2.COLOR_GRAY2BGR)
    #看圆的半径，得出型号，一个为109，一个为208,208的字符高度为50左右，109的为80左右
    code_high=0
    code_size=0
    if(circle_r>204):
        h_thresh=42
        code_high = 22
        code_size = 1
    else:
        h_thresh=78
        code_high = 25
        code_size = 1.5
    fun='svm'
    import 第四章.模板匹配算法.formated.模板匹配识别字符 as detect
    import 第四章.svm.svm识别字符模型训练 as svm_detect
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if(area>100):
            x, y, w, h = cv2.boundingRect(contours[i])
            if(h>h_thresh):
                count+=1
                cv2.rectangle(unflood_srcImg,(x-3,y-3),(x+w+3,y+h+3),color=(255,0,0))
                cutNum=0
                # tempImg=unflood_srcImg[x-cutNum:x+w+cutNum,y-cutNum:y+h+cutNum,:]
                tempImg=unflood_srcImg[y-cutNum:y+h+cutNum,x-cutNum:x+w+cutNum,:]
                tempImg=cv2.cvtColor(tempImg,cv2.COLOR_RGB2GRAY)
                if fun=='moduel':
                    ret_code=detect.detect_code(tempImg,show_plt=False)
                    unflood_srcImg = cv2.putText(unflood_srcImg, str(ret_code), (x, y + code_high), cv2.FONT_HERSHEY_COMPLEX,code_size, (255, 0, 0), 2)
                if fun=='svm':
                    # if(tempImg.width>0):
                        ret_code=svm_detect.dettect_code(tempImg)
                        unflood_srcImg=cv2.putText(unflood_srcImg,str(ret_code),(x,y+code_high),cv2.FONT_HERSHEY_COMPLEX,code_size,(255,0,0),2)

    plt.figure()
    plt.imshow(unflood_srcImg)
    plt.title('字符定位,共{0}个'.format(str(count)), fontsize=_font_size)
    plt.show()


if __name__=='__main__':
    img=cv2.imread('OKPictures\\17_34_06.jpg',cv2.IMREAD_GRAYSCALE)
    last_fun(img)