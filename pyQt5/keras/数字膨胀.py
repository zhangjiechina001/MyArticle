import cv2

src=cv2.imread('num_8.jpg',cv2.IMREAD_GRAYSCALE)
kernal=cv2.getStructuringElement(1,(3,3))
src=cv2.dilate(src,kernel=kernal)
# src=cv2.erode(src,kernel=kernal)
cv2.imshow('dilate',src)
cv2.waitKey()
cv2.imwrite('num_888.jpg',src)