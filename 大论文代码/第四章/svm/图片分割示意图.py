import cv2
import matplotlib.pyplot as plt
import numpy as np

def create_cut_img(img):
    count=0
    for i in range(30):
        for j in range(10):

            if(i%3==0):
                count = count + 1
                tempImg=img[j:j+85,i+10:i+60]
                plt.subplot(10,10,count)
                plt.xticks([])
                plt.yticks([])
            plt.imshow(tempImg,cmap='gray')

    plt.show()

src=cv2.imread('formatedImg\\8.jpg',cv2.IMREAD_GRAYSCALE)
create_cut_img(src)
