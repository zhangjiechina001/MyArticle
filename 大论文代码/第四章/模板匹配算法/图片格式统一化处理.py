import cv2
import numpy as np

def formatImg(file_name):
    img=cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    h,w=img.shape
    mask=np.zeros((100,100),dtype='u1')
    mask[(100-h)//2:(100-h)//2+h,(100-w)//2:(100-w)//2+w]=img
    return mask

if __name__=='__main__':
    import os
    files=os.listdir(r'E:\大论文\大论文代码\第四章\模板匹配算法\cutedImg')
    for file in files:
        format=formatImg(file_name='cutedImg\\'+file)
        cv2.imshow(file,format)
        cv2.imwrite(str(file),format)
    print('success!!!')