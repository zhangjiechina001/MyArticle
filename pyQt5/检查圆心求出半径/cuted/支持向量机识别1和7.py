from sklearn.datasets import load_digits
import numpy as np
import cv2

#将数字裁剪，制作训练集
#共有120个1标签，60个7标签
X_train=np.ones((150,3844),dtype=np.float32)
Y_train=np.zeros(150,dtype=np.uint8)
X_test=np.ones((30,3844),dtype=np.float32)
Y_test=np.zeros(30,dtype=np.uint8)
#读取1的训练集以及标签
for i in range(120):
    temp=cv2.imread('num1\\traindata\\'+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)
    ravel=temp.ravel()
    if i<100:
        #0-99
        X_train[i,:]=ravel
        Y_train[i]=1
    else:
        #0-19
        X_test[i-100,:]=ravel
        Y_test[i-100]=1

for i in range(60):
    temp=cv2.imread('num7\\traindata\\'+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)
    ravel=temp.ravel()
    if i<50:
        X_train[i+100,:]=ravel
        Y_train[i+100]=7
    else:
# 20-30
        X_test[i+20-50,:]=ravel
        Y_test[i+20-50]=7

print("训练数据量:",Y_train.shape)
print("测试数据量:",Y_test.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)
lsvc=LinearSVC()
lsvc.fit(X_train,Y_train)
y_prtect=lsvc.predict(X_test)
#读取数据，预测值
x_test=cv2.imread(r'num1\traindata\6.jpg',cv2.IMREAD_GRAYSCALE)
x_test=x_test.ravel()
pertect=lsvc.predict([x_test])
print(y_prtect)
print(Y_test)
print(pertect)

