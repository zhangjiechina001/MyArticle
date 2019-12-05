import os

import cv2
import numpy as np
from sklearn.datasets import load_digits


# files=os.listdir(r'traindata\Sample011')
# 读取数据，返回的是数组和标签
def readSingalData(directory, label):
    files = os.listdir(directory)
    returnDatas, returnlabels = [], []
    for file in files:
        temp = cv2.imread(directory + '\\' + file, cv2.IMREAD_GRAYSCALE)
        temp = cv2.pyrDown(temp)
        temp = cv2.pyrDown(temp)
        # temp = cv2.pyrDown(temp)
        # temp = cv2.pyrDown(temp)
        # temp = cv2.pyrDown(temp)
        # temp = cv2.pyrDown(temp)
        temp = temp.ravel()
        returnlabels.append(label)
        returnDatas.append(temp)
    return returnDatas, returnlabels


def directoryIni():
    dirtoryArr = []
    for i in range(26):
        dir = 'traindata\\Sample0' + str(i + 11)
        dirtoryArr.append(dir)
    return dirtoryArr


# 训练数据集的准备
def ReadTraindata():
    directoryArr = directoryIni()
    labelsArr = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R'
        , 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(len(directoryArr)):
        tempdatas, templabels = readSingalData(directoryArr[i], labelsArr[i])
        X_train.append(tempdatas)
        X_test.append(templabels)
    return X_train, X_test


X_train, y_train = ReadTraindata()
X_train = np.reshape(X_train, (26 * 55, 1080000 // 16))
y_train = np.reshape(y_train, (26 * 55))
print("数据集量及单个数据的大小：", len(X_train), len(y_train))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
predict = cv2.imread(r'traindata\Sample011\img011-001.png', cv2.IMREAD_GRAYSCALE)
predict = cv2.pyrDown(predict)
predict = cv2.pyrDown(predict)
# predict=cv2.pyrDown(predict)
# predict=cv2.pyrDown(predict)
# predict=cv2.pyrDown(predict)
# predict=cv2.pyrDown(predict)
cv2.imshow('E', predict)
cv2.waitKey()
predict = ss.fit_transform([predict.ravel()])
y_pretect = lsvc.predict(predict)
print(y_pretect)

# digits=load_digits('EnglishHnd.tgz')
# print("数据集量及单个数据的大小：",digits.data.shape)
#
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.5,random_state=33)
# # X_train=np.resize(X_train,(8,8))
# # y_train=np.resize(y_train,(8,8))
# print("训练数据量:",y_train.shape)
# print("测试数据量:",y_test.shape)
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import LinearSVC
#
#
# pretect=cv2.imread('5.png',cv2.IMREAD_GRAYSCALE)
# keral=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
# pretect=cv2.erode(pretect,keral)
# print(pretect.shape)
# pretect=cv2.bitwise_not(pretect)
# pretect=cv2.resize(pretect,(8,8))
# cv2.imshow('pretect',pretect)
# cv2.imwrite('pretect.jpg',pretect)
# cv2.waitKey()
#
# num_1=np.array([pretect.ravel()],np.float)
# cv2.imshow('pretect',pretect)
# # num_4=cv2.imread('num_4.png',cv2.IMREAD_GRAYSCALE)
# # cv2.imshow('test')
# X_test=ss.transform(X_test)
#
# lsvc=LinearSVC()
# lsvc.fit(X_train,y_train)
# y_predict=lsvc.predict(X_test)
# y_testpredict=lsvc.predict(num_1)
# print(str(y_testpredict[0]))
# print('The accuracy of Linear SVC is:',lsvc.score(X_test,y_test))
# from sklearn.metrics import classification_report
# classification=classification_report(y_test,y_predict, target_names=digits.target_names.astype(str))
# print(classification)
