from sklearn.datasets import load_digits
import numpy as np
import cv2
digits=load_digits()
print("数据集量及单个数据的大小：",digits.data.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.5,random_state=33)
# X_train=np.resize(X_train,(8,8))
# y_train=np.resize(y_train,(8,8))
print("训练数据量:",y_train.shape)
print("测试数据量:",y_test.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
ss=StandardScaler()
X_train=ss.fit_transform(X_train)


pretect=cv2.imread('5.png',cv2.IMREAD_GRAYSCALE)
keral=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
pretect=cv2.erode(pretect,keral)
print(pretect.shape)
pretect=cv2.bitwise_not(pretect)
pretect=cv2.resize(pretect,(8,8))
cv2.imshow('pretect',pretect)
cv2.imwrite('pretect.jpg',pretect)
cv2.waitKey()

num_1=np.array([pretect.ravel()],np.float)
cv2.imshow('pretect',pretect)
# num_4=cv2.imread('num_4.png',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('test')
X_test=ss.transform(X_test)

lsvc=LinearSVC()
lsvc.fit(X_train,y_train)
y_predict=lsvc.predict(X_test)
y_testpredict=lsvc.predict(num_1)
print(str(y_testpredict[0]))
print('The accuracy of Linear SVC is:',lsvc.score(X_test,y_test))
from sklearn.metrics import classification_report
classification=classification_report(y_test,y_predict, target_names=digits.target_names.astype(str))
print(classification)
