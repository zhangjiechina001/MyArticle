import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
ax=plt.subplot(1,1,1,projection='3d')
import numpy as np
x=np.linspace(0,10,100)
y=np.linspace(-10,0,100)
z=np.sin(x+y)
ax.plot(x,y,z)
