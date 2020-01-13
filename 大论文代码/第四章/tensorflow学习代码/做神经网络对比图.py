import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def draw_bar():
    label_list = ['10分类', '2分类', '26分类', '36分类']    # 横坐标刻度显示值
    num_list1 = [99.93, 99.73, 95.7, 94.57]      # 纵坐标值1
    num_list2 = [100.00, 100.00, 100.00, 96.68]      # 纵坐标值2
    x = list(range(0,4))
    """
    绘制条形图
    left:长条形中点横坐标
    height:长条形高度
    width:长条形宽度，默认值0.8
    label:为后面设置legend准备
    """
    rects1 = plt.bar(x=x, height=num_list1, width=0.4, alpha=0.8, color='red', label="测试集")
    rects2 = plt.bar(x=[i + 0.4 for i in x], height=num_list2, width=0.4, color='g', label="训练集")
    plt.ylim(90, 102)     # y轴取值范围
    plt.ylabel("准确率(%)")
    """
    设置x轴刻度显示值
    参数一：中点坐标
    参数二：显示值
    """
    plt.xticks([index + 0.2 for index in x], label_list)
    plt.xlabel("分类数")
    plt.title("数字字母分类情况")
    plt.legend()     # 设置题注
    # 编辑文本
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+0.1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+0.1, str(height), ha="center", va="bottom")

plt.figure()
plt.subplot(2,2,1)

img=plt.imread('tensorboard_train _images/2_classfication.png')
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.title("2分类", fontsize=20)
plt.subplot(2,2,2)

img=plt.imread('tensorboard_train _images/10_classfication.png')
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.title("10分类", fontsize=20)
plt.subplot(2,2,3)

img=plt.imread('tensorboard_train _images/26_classfication.png')
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.title("26分类", fontsize=20)

plt.subplot(2,2,4)
draw_bar()
plt.show()