import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from scipy import signal

#创建一个class，包括识别的所有功能
#两次缩小=>得到二值图=>提取圆心半径=>输入图片，圆心点，展开内半径，展开宽度
#=>返回计算极值，开始角度，卷积结果
class recongnize_image():
    def __init__(self):
        print('recongnize_image init success!')

    #两次缩小
    def reduceImg(self,img):
        ret_img=cv2.pyrDown(img)
        ret_img=cv2.pyrDown(ret_img)
        return ret_img

    # 得到二值图
    def binaryImage(self,img, binary_type):
        ret, binary = cv2.threshold(img, 141, 255, binary_type)
        return ret, binary

    # 提取圆心，半径
    def getPointAndR(self,src):
        _, contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result_img = np.zeros(src.shape, np.uint8)
        retImg = None
        i = 1
        circle_point = None
        circle_r = None
        # for contour in contours:
        #     area=cv2.contourArea(contour)
        #     print('area{0}:{1}'.format(i,str(area)))
        #     i+=1
        resizeNum = 16

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            print(area)
            if (area < 1500000 // resizeNum or area > 2500000 // resizeNum):
                continue
            x, y, w, h = cv2.boundingRect(contours[i])
            ratio = 0.0
            if (y != 0):
                ratio = float(w / h)
            if ((ratio > 0.95) & (ratio < 1.05)):
                cv2.drawContours(result_img, contours, i, (255, 255, 255), thickness=5)
                # cv2.namedWindow('cut',cv2.WINDOW_NORMAL)
                # cv2.imshow('cut',result_img)
                # if(area>2000000 and area<2500000):
                size = 100
                # else:
                #     size=0
                # retImg = readImg[y - size:y + h + size, x - size:x + w + size]
                circle_point = (y + w // 2, x + h // 2)
                circle_r = w // 2
                break
        return circle_point, circle_r, result_img

    def unfloodImagePro(self,img, point, unfloodR, width, startTheta):
        h, w = img.shape
        x0, y0 = point
        unwrapped_width = unfloodR + width  # 展开的最大半径
        unwrapped_height = width
        full_width = int(2 * math.pi * unwrapped_width)  # 展开后的长度
        unflood_width = 950
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

    # 输入图片，圆心点，展开内半径，展开宽度
    def unfloodImage(self,img, point, unfloodR, width, startTheta):
        h, w = img.shape
        x0, y0 = point
        unwrapped_width = unfloodR + width  # 展开的最大半径
        unwrapped_height = width
        full_width = int(2 * math.pi * unwrapped_width)  # 展开后的长度
        unwrapped_img = np.zeros((unwrapped_height, full_width), dtype='u1')
        except_count = 0
        for j in range(full_width):
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

    # 返回计算极值，开始角度，卷积结果
    def calcMax(self,img):
        h, w = img.shape
        kernel = np.ones((h, 230), dtype=np.float32)
        # kernel[0:10,:]=0
        result = signal.convolve2d(img, kernel, 'valid')
        hist = result.ravel()
        ret_max = np.max(hist)
        start_position = np.argmax(hist)
        # start_position=start_position[0]
        ret_theta = (start_position / w) * 2 * np.pi
        return ret_max, ret_theta, hist

    def seamlessCone(self,im, obj):
        # # Read images : src image will be cloned into dst
        # im = cv2.imread("bg_img.png")
        # obj = cv2.imread("char_1.jpg")

        # Create an all white mask
        mask = 255 * np.ones(obj.shape, obj.dtype)

        # The location of the center of the src in the dst
        width, height, channels = im.shape
        center = (height // 2, width // 2)

        # Seamlessly clone src into dst and put the results in output
        normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
        mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)

        # Write results
        # cv2.imshow("opencv-normal-clone-example.jpg", normal_clone)
        # cv2.imshow("pencv-mixed-clone-example.jpg", mixed_clone)
        # cv2.waitKey()
        return mixed_clone

    def pic_mix(self,im, obj):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        h, w = obj.shape
        im[(100 - h) // 2:(100 - h) // 2 + h, (80 - w) // 2:(80 - w) // 2 + w] = obj[:, :]
        im = cv2.GaussianBlur(im, (5, 5), 0)
        return im

    j = 0

    def disImgs(self,img_list):
        plt.figure()
        im = cv2.imread('recongnize_history/bg_img.png')
        for i in range(12):
            # img_list[i] = cv2.pyrDown(img_list[i])
            # w,h=img_list[i].shape
            # tempplate[:,:]=255
            # img_list[i]=cv2.pyrDown(img_list[i])
            # mask=255*np.ones(img_list[i].shape,dtype=img_list[i].dtype)
            # mask[(100 - h) // 2:(100 - h) // 2 + h, (80 - w) // 2:(80 -w ) // 2 + w]=255
            # tempplate[(100 - h) // 2:(100 - h) // 2 + h, (80 - w) // 2:(80 -w ) // 2 + w] = img_list[i][:, :]
            # img_list[i]=cv2.resize(tempplate,(26,26))
            # tempplate[(100 - h) // 2:(100 - h) // 2 + h, (80 - w) // 2:(80 -w ) // 2 + w]=255
            plt.subplot(3, 4, i + 1)
            # 85,24
            obj = img_list[i]
            # cv2.imwrite('char_1.jpg',src_1)
            # res=seamlessCone(im,obj)
            res = self.pic_mix(im, obj)
            res = cv2.resize(res, (28, 28))
            plt.imshow(res, cmap='gray')

    def last_fun(self,img):
        copyImg = img.copy()
        # 两次缩小
        smallImg = self.reduceImg(img)
        plt.subplot(1, 3, 1)
        plt.imshow(smallImg, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title("两次缩小图", fontsize=10)
        # 二值化
        thresh, binary = binaryImage(smallImg, binary_type=cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        plt.subplot(1, 3, 2)
        plt.imshow(binary, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title("二值化阈值：{0}".format(str(thresh)), fontsize=10)
        # 提取圆心，半径
        circle_point, circle_r, drawImg = getPointAndR(binary)
        plt.subplot(1, 3, 3)
        plt.imshow(drawImg, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title("圆心:{0},半径：{1}".format(str(circle_point), str(circle_r)), fontsize=10)
        # 分角度展开灰度图
        # 1.展开角为0
        plt.figure()
        unflood_img1 = unfloodImage(smallImg, point=circle_point, unfloodR=circle_r, width=40, startTheta=0)
        unflood_img1 = cv2.Canny(unflood_img1, 100, 200)
        thresh = 0
        plt.subplot(4, 1, 1)
        plt.imshow(unflood_img1, cmap='gray')
        # plt.xticks([])
        # plt.yticks([])
        plt.title("第一次展开,thresh:{0}".format(str(thresh)), fontsize=20)
        # 绘制直方分布图
        plt.subplot(4, 1, 2)
        ret_max1, ret_theta1, hist1 = calcMax(unflood_img1)
        plt.plot(hist1)
        plt.title('max:{0},theta:{1}'.format(str(ret_max1), str(ret_theta1)), fontsize=20)

        # 2.展开角为90°
        unflood_img2 = unfloodImage(smallImg, point=circle_point, unfloodR=circle_r, width=40, startTheta=math.pi / 2)
        unflood_img2 = cv2.Canny(unflood_img2, 100, 200)
        plt.subplot(4, 1, 3)
        plt.imshow(unflood_img2, cmap='gray')
        # plt.xticks([])
        # plt.yticks([])
        plt.title("第二次展开,thresh:{0}".format(str(thresh)), fontsize=20)

        # 绘制直方分布图
        plt.subplot(4, 1, 4)
        ret_max2, ret_theta2, hist2 = calcMax(unflood_img2)
        plt.plot(hist2)
        plt.title('max:{0},theta:{1}'.format(str(ret_max2), str(ret_theta2 + math.pi / 2)), fontsize=20)

        _font_size = 20
        # 确定展开角度
        plt.figure()
        plt.subplot(4, 1, 1)
        # theta=0.0
        if (ret_max1 > ret_max2):
            theta = ret_theta1
        else:
            theta = ret_theta2 + math.pi / 2
        # 确定大图展开位置
        # circle_point, circle_r, drawImg
        point = (circle_point[0] * 4, circle_point[1] * 4)
        r = circle_r * 4

        # 看圆的半径，得出型号，一个为109，一个为208,208的字符高度为50左右，109的为80左右
        unflood_srcImg = unfloodImagePro(copyImg, point=point, unfloodR=r, width=130, startTheta=theta - 0.03)
        if (circle_r > 204):
            unflood_srcImg = unflood_srcImg[40:110, :]
        else:
            unflood_srcImg = unflood_srcImg[0:100, :]

        plt.imshow(unflood_srcImg, cmap='gray')
        plt.title('原图展开{0}'.format(str(theta)), fontsize=_font_size)

        plt.subplot(4, 1, 2)
        binaryUnfloodImg = cv2.adaptiveThreshold(unflood_srcImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                                 blockSize=17, C=10)
        # cv2.imshow('img',tempimg)
        # binaryUnfloodImg=binaryImage(unflood_srcImg,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        plt.imshow(binaryUnfloodImg, cmap='gray')
        plt.title('局部自适应二值化', fontsize=_font_size)

        plt.subplot(4, 1, 3)
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dst_img = binaryUnfloodImg
        # CLOSE先膨胀再腐蚀
        dst_img = cv2.morphologyEx(dst_img, cv2.MORPH_CLOSE, kernel)
        # OPEN先腐蚀再膨胀
        dst_img = cv2.morphologyEx(dst_img, cv2.MORPH_OPEN, kernel)
        # dst_img=cv2.Canny(dst_img,100,200)
        plt.imshow(dst_img, cmap='gray')
        plt.title('形态学开操作:kernel={0}'.format(str((2, 2))), fontsize=_font_size)

        plt.subplot(4, 1, 4)
        # 字符切割
        _, contours, _ = cv2.findContours(dst_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        unflood_srcImg = cv2.cvtColor(unflood_srcImg, cv2.COLOR_GRAY2BGR)
        temp_srcImg = unflood_srcImg.copy()
        # 看圆的半径，得出型号，一个为109，一个为208,208的字符高度为50左右，109的为80左右
        code_high = 0
        code_size = 0
        if (circle_r > 204):
            h_thresh = 42
            code_high = 22
            code_size = 1
        else:
            h_thresh = 78
            code_high = 25
            code_size = 1.5
        fun = 'module'
        import 第四章.模板匹配算法.formated.模板匹配识别字符 as detect
        import 第四章.svm.svm识别字符模型训练 as svm_detect
        img_list = []
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if (area > 100):
                x, y, w, h = cv2.boundingRect(contours[i])
                if (h > h_thresh):
                    count += 1
                    unflood_srcImg_copy = unflood_srcImg.copy()
                    cv2.rectangle(unflood_srcImg, (x - 3, y - 3), (x + w + 3, y + h + 3), color=(255, 0, 0))
                    cutNum = 0
                    # tempImg=unflood_srcImg[x-cutNum:x+w+cutNum,y-cutNum:y+h+cutNum,:]
                    tempImg = unflood_srcImg[y - cutNum:y + h + cutNum, x - cutNum:x + w + cutNum, :]
                    tempImg = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
                    cutNum = 3
                    tempImg_1 = unflood_srcImg_copy[y - cutNum:y + h + cutNum, x - cutNum:x + w + cutNum, :]
                    tempImg_1 = cv2.cvtColor(tempImg_1, cv2.COLOR_RGB2GRAY)
                    img_list.append(tempImg_1)
                    if fun == 'module':
                        ret_code = detect.detect_code(tempImg, show_plt=False)
                        unflood_srcImg = cv2.putText(unflood_srcImg, str(ret_code), (x, y + code_high),
                                                     cv2.FONT_HERSHEY_COMPLEX, code_size, (255, 0, 0), 2)
                    if fun == 'svm':
                        # if(tempImg.width>0):
                        ret_code = svm_detect.dettect_code(tempImg)
                        # unflood_srcImg=cv2.putText(unflood_srcImg,str(ret_code),(x,y+code_high),cv2.FONT_HERSHEY_COMPLEX,code_size,(255,0,0),2)
        disImgs(img_list)
        plt.figure()
        plt.imshow(unflood_srcImg)
        import time
        # time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        pic_name = 'recongnize_history/' + time.strftime("%Y-%m-%d %H_%M_%S", time.localtime()) + '.jpg'
        # cv2.imwrite(pic_name,unflood_srcImg)
        plt.title('字符定位,共{0}个'.format(str(count)), fontsize=_font_size)
        plt.show()
