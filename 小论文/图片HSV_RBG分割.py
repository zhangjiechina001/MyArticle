import cv2

src=cv2.imread('09_30_13.jpg')
splitType=['h','s','v']
hsv=src
hsv=cv2.split(hsv)
for i in range(len(splitType)):
    cv2.namedWindow(splitType[i],cv2.WINDOW_NORMAL)
    cv2.imshow(splitType[i],hsv[i])

cv2.waitKey()
cv2.destroyAllWindows()
