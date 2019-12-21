#将读取的照片存在csv文件中
import pandas as pd
import cv2
import numpy as np
data_names='0,1,2,4,6,7,8,9,A,B'.split(',')
def read_img():
    for name in data_names:
        src = cv2.imread('formatedImg//'+str(name)+'.jpg', cv2.IMREAD_GRAYSCALE)
        _, src = cv2.threshold(src, 10, 1, cv2.THRESH_BINARY)
        src=cv2.pyrDown(src)
        for i in range(10):
            for j in range(5):
                img=src[j:j+45,i:i+40]
                temp=np.ones([50,50])
                h,w=img.shape
                temp[(50-h)//2:(50-h)//2+h,(50-w)//2:(50-w)//2+w]=img
                temp=temp.reshape([1,50*50])
                # temp=np.zeros([1,w*h+1])
                # temp[0,0:w*h]=
                data=pd.DataFrame(temp)
                data['label']=name
                data.to_csv('data.csv',mode='a',header=0)

read_img()