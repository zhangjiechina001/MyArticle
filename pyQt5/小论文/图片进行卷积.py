import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
import hmmlearn

def CutNumArea(img):
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    img=cv2.erode(img,kernel)
    h, w = img.shape


    areas=3
    plt.subplot2grid((areas,6),(0,0),colspan=6)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))
    plt.xlabel('(a)原图', fontproperties='SimHei', fontsize=35)
    plt.xticks([])
    plt.yticks([])

    plt.subplot2grid((areas,6),(1,0),colspan=6)
    kernel = np.ones((h, 15), dtype=np.float32)
    result = signal.convolve2d(img, kernel, 'valid')
    # result = cv2.threshold(result, 1000, 1, cv2.THRESH_BINARY)
    hist = result.ravel()
    plt.plot(hist)
    plt.xlabel('(b)卷积投影分布', fontproperties='SimHei', fontsize=35)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # img = cv2.erode(img, kernel)
    # 先将大的分割出来
    plt.subplot2grid((areas,6),(2,0),colspan=6)
    kernel = np.ones((h, 15), dtype=np.float32)
    result = signal.convolve2d(img, kernel, 'valid')
    result = cv2.threshold(result, 1, 1000, cv2.THRESH_BINARY)
    hist = result[1].ravel()
    plt.plot(hist)
    plt.xlabel('(c)阀值处理后分布', fontproperties='SimHei', fontsize=35)

    startIdx = []
    endIdx = []
    for i in range(1,len(hist)):
        if (hist[i - 1] == 1) & (hist[i] == 0):
            endIdx.append(i)
        if (hist[i - 1] == 0) & (hist[i] == 1):
            startIdx.append(i)
    startIdx.insert(0,0)
    endIdx.append(w)
    # for i in range(len(startIdx)):
    #     if(endIdx[i]-startIdx[i]>50):
    #         mediu=(startIdx[i]+endIdx[i])//2
    #         startIdx.insert(i+1,mediu)
    #         endIdx.insert(i,mediu)
    imgarr = []
    for i in range(len(startIdx)):
        tempimg = img[:, startIdx[i]:endIdx[i]]
        imgarr.append(tempimg)
    mask=np.ones(shape=(h,w),dtype=np.uint8)
    _,mask=cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    count=5
    # for i in range(len(imgarr)):
    #     plt.subplot2grid((4, 6), (3, i), rowspan=1,colspan=1)
    #     plt.imshow(cv2.cvtColor(imgarr[i],cv2.COLOR_GRAY2RGB))
    #     # plt.figure(figsize=(10,10))
    #     plt.xlabel(str.format('(d){0}',str(i+1)), fontproperties='SimHei', fontsize=20)
    #     plt.xticks([])
    #     plt.yticks([])
    plt.show()

    return imgarr

#对原图片进行卷积，得出定位点
dst_img=cv2.imread('aa.jpg',cv2.IMREAD_GRAYSCALE)
CutNumArea(dst_img)

plt.show()
cv2.waitKey()
cv2.destroyAllWindows()