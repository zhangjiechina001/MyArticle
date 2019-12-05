import  cv2
import numpy as np


src=cv2.imread('cimg_1_4.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('src',src)
cv2.waitKey()
cv2.destroyAllWindows()

for i in range(30):
    mask = np.zeros((62, 62), np.uint8)
    for j in range(29):
        mask[:,i+j]=src[:,j]
        cv2.imshow(str(i),mask)
        cv2.imwrite('traindata\\'+str(i+30)+'.jpg',mask)
cv2.waitKey()
cv2.destroyAllWindows()