import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
img=mpimg.imread('bestimg.jpg')
ax1=plt.subplot(121)
ax1.imshow(img)
plt.imshow(img)
ax2=plt.subplot(122)
ax2.hist(img.ravel(),bins=1000)

plt.show()