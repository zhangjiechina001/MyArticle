import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

# def metric(fn):
#     print('%s executed in %s ms' % (fn.__name__, 10.24))
#     return fn
# @metric
# def customize(a,b):
#     print("我是被装饰的函数,运行结果%d"%(a+b))
# customize(1,3)
def blursImg(imgInfoList):
    plt.figure()
    i=0
    for imginfo in imgInfoList:
        blurName,imgData,dealTime=imginfo
        i=i+1
        plt.subplot(2,3,i)
        plt.imshow(imgData)
        plt.xticks([])
        plt.yticks([])
        tempyime="%0.3f" % dealTime
        plt.title("{0}({1}ms)".format(blurName,str(tempyime)), fontsize=10)
    plt.show()