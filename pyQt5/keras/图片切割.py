import cv2

src=cv2.imread('0__10.jpg',cv2.IMREAD_GRAYSCALE)
src=src[:,0:15]
cv2.imshow('tempNum5',src)
cv2.imwrite('num_8.jpg',src)
cv2.waitKey()
print('save success!')