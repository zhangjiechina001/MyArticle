import cv2
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=10,ncols=10,sharex='all',sharey='all')
ax = ax.flatten()
for i in range(11):
    name='getImages_format/'+str(i+1)+'.jpg'
    img=cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    ax[i].imshow(img,cmap='gray')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()