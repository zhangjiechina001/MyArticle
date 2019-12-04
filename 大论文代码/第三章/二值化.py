import cv2 as cv
def threshold_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)   #要二值化图像，要先进行灰度化处理
    ret, binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value: %s"%ret)#打印阈值，前面先进行了灰度处理0-255，我们使用该阈值进行处理，低于该阈值的图像部分全为黑，高于该阈值则为白色
    cv.imshow("binary",binary)#显示二值化图像
    cv.waitKey()

def local_threshold(image):
    gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)   #要二值化图像，要先进行灰度化处理
    dst = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,25,10)
    cv.imshow("local_threshold", dst);cv.waitKey()

img=cv.imread('09_30_13.jpg')
# threshold_demo(img)
local_threshold(img)