from tokenize import String

from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
dataset_size=100

rdm=RandomState(1)
X=rdm.random_sample([dataset_size,2])
X=np.concatenate((X,np.array([[1.2,1.2],[1.15,1.2]])),axis=0)
plt.scatter(X[:,0],X[:,1],marker='^')
rdm=RandomState(2)
X=rdm.random_sample([dataset_size,2])
X1=X+1.1
X1=np.concatenate((X1,np.array([[0.9,0.95],[0.93,0.95]])),axis=0)
plt.scatter(X1[:,0],X1[:,1],marker='o')

c=np.concatenate((X,X1),axis=0)
Y=[[0 if x1+x2<2 else 1] for (x1,x2) in c]
clf=svm.SVC(kernel='rbf')
clf.fit(c,Y)
result=clf.predict([[1,0.5]])
print(result)

draw_x=np.linspace(0,2.1,200)
y=-1.3*draw_x+2.1
y1=-1.3*draw_x+2.2
y2=-1.3*draw_x+2.0
plt.plot(draw_x,y)
plt.plot(draw_x,y1,linestyle=':')
plt.plot(draw_x,y2,linestyle=':')
plt.show()
