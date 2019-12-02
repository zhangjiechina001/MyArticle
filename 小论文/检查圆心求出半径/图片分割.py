import cv2

src=cv2.imread('bestimg.jpg')
src=src[14:74,138:172,]
cv2.imshow('src',src)
cv2.waitKey()
cv2.imwrite('cut1.jpg',src)
cv2.destroyAllWindows()