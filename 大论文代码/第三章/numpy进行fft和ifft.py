"""
使用 numpy 进行 FFT 和 IFFT
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
#读取照片
img = cv2.imread('09_30_13.jpg', 0)
#傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))


rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)
tempnum=100
magnitude_spectrum[crow - tempnum: crow + tempnum, ccol - tempnum: ccol + tempnum] = 0

fshift[crow - tempnum: crow + tempnum, ccol - tempnum: ccol + tempnum] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

"""
使用 OpenCV 进行 FFT 和 IFFT
"""
# cv2.dft( )函数返回值是双通道的，第一个通道是实数部分，第二个是虚数部分
# cv2.dft( )函数输入图像要先转为 np.float32 格式
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum2 = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# 创建低通滤波器掩模、使用掩模滤波、IDFT
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
tempmask=np.zeros((rows, cols), np.uint8)
tempnum=100
tempmask[crow - tempnum:crow + tempnum, ccol - tempnum:ccol + tempnum] = 1
magnitude_spectrum2=magnitude_spectrum2*tempmask
f_dft_shift = dft_shift * mask
f_idft_shift = np.fft.ifftshift(f_dft_shift)

img_back2 = cv2.idft(f_idft_shift)
img_back2 = cv2.magnitude(img_back2[:, :, 0], img_back2[:, :, 1])

# 显示结果图像
imgList = [img, magnitude_spectrum, img_back, img, magnitude_spectrum2, img_back2]
imgName = ['原图', '高通滤波', '效果图', '原图', '低通滤波', '效果图']

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(imgList[i], cmap='gray'), plt.title(imgName[i],fontsize=25)
    plt.xticks([]), plt.yticks([])
plt.show()

