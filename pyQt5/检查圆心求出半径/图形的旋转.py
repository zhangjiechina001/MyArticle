import cv2
# src=cv2.imread('imgSave.jpg',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('original',src)
# x,y=src.shape[:2]
# rotate = cv2.getRotationMatrix2D((x/2,y/2),45,0.5)
# res = cv2.warpAffine(src,rotate,(0.5*x,0.5*y))
# cv2.imshow('rotate',res)
# cv2.waitKey()
# cv2.destroyAllWindows()
import cv2
img1 = cv2.imread('imgSave.jpg', cv2.IMREAD_COLOR)
print(img1.shape)
x, y = img1.shape[:2]
rotate = cv2.getRotationMatrix2D((x / 2, y / 2), -150, 1)
res = cv2.warpAffine(img1, rotate, (int( x), int( y)))
print(res.shape)
cv2.imshow('img', img1)
cv2.namedWindow('res',cv2.WINDOW_NORMAL)
cv2.imshow('res', res)
cv2.waitKey(0)
