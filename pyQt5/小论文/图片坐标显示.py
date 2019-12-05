import matplotlib.image as imp
import matplotlib.pyplot as plt

src=imp.imread('afterCutImg.png')
plt.imshow(src)
plt.show()
import cv2
print(cv2.__version__)