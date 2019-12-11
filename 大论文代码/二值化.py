import cv2
src=cv2.imread('imgSave.jpg',cv2.IMREAD_GRAYSCALE)
_,binary=cv2.threshold(src,143,255,cv2.THRESH_BINARY_INV)
cv2.imwrite('unflood_binary.png',binary)