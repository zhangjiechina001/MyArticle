import cv2

src=cv2.imread('bestimg.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('grayscale',src)
h,w=src.shape
src1=src[:,0:300]
src2=src[:,300:w]
cv2.imshow('src1',src1)
cv2.imshow('src2',src2)



cv2.waitKey()
cv2.destroyAllWindows()