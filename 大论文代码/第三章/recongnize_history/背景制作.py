import  cv2
import numpy as np
src=cv2.imread('2019-12-27 10_54_32.jpg',cv2.IMREAD_GRAYSCALE)
src_img=np.zeros(shape=(100,80),dtype=np.uint8)
src_img[:,0:40]=src[:,0:40]
src_img[:,40:80]=src[:,0:40]

cv2.imshow('result_image',src_img)
cv2.imwrite('bg_img.png',src_img)
cv2.waitKey()