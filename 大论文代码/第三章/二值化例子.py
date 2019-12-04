import cv2 as cv


# 图像二值化 0白色 1黑色
def threshold_image(image):
    # 以灰度化形式读取图片
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # OpenCV中的窗口无法显示中文，未解决
    # OpenCV中读取文件路径也不能包含中文，否则会出错
    cv.imshow("original", gray)
    # OpenCV中读取的图像形式是以数组的形式读取与PIL中的image模块的读取不一样
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 大律法,全局自适应阈值 参数0可改为任意数字但不起作用
    print("大律法阈值：%s" % ret)
    cv.imshow("OTSU", binary)
    # ret可输出该阈值类型下进行二值化所选取的阈值
    ret, binary = cv.threshold(gray, 0, 255,
                               cv.THRESH_BINARY | cv.THRESH_TRIANGLE)  # TRIANGLE法,，全局自适应阈值, 参数0可改为任意数字但不起作用，适用于单个波峰
    print("阈值：%s" % ret)
    cv.imshow("TRIANGLE", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)  # 自定义阈值为150,大于150的是白色 小于的是黑色
    print("阈值：%s" % ret)
    cv.imshow("1", binary)

    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)  # 自定义阈值为150,大于150的是黑色 小于的是白色
    print("阈值：%s" % ret)
    cv.imshow("2", binary)


src = cv.imread("09_30_13.jpg")
threshold_image(src)
cv.waitKey(0)
cv.destroyAllWindows()
