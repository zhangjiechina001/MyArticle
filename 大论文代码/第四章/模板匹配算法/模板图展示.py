import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
name_list='0，1，2，4，6，7，8，9，A，B'.split('，')

def display_mould_pic():
    i=0
    for name in name_list:
        i+=1
        img=cv2.imread('formated\\cut_format\\'+name+'.jpg',cv2.IMREAD_GRAYSCALE)
        plt.subplot(2, 5, i )
        plt.title('字符{0}'.format(name),fontsize=20)
        plt.imshow(img, cmap='gray')
    plt.show()

if __name__=='__main__':
    display_mould_pic()