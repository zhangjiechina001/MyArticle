from numpy.random import RandomState
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
import numpy as np
dataset_size=200
rdm=RandomState(1)
X=rdm.random_sample([dataset_size,2])
colors1 = '#00CED1' #点的颜色
colors2 = '#DC143C'
# plt.subplot(1,2,1)
for x in X:
    a,b=x
    if(((a-0.5)**2+(b-0.5)**2)<0.16):
        plt.scatter(a, b, marker='o',c='r')
    else:
        plt.scatter(a, b, marker='^',c='b')



# plt.subplot(1,2,2)

fig = plt.figure()
ax = Axes3D(fig)
for x in X:
    a,b=x
    R = (a-0.5)**2+(b-0.5)**2
    if(((a-0.5)**2+(b-0.5)**2)<0.16):
        ax.scatter(a, b,R, marker='o',c='r')
    else:
        ax.scatter(a, b,R, marker='^',c='b')

X = np.arange(0, 1, 0.1)
Y = np.arange(0, 1, 0.1)
X, Y = np.meshgrid(X, Y)
R = np.zeros([10,10])
R[:,:]=0.16
# Z = np.sin(R)
ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap=cm.viridis)


plt.show()
