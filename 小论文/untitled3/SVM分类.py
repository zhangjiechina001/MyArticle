#svm本质是寻求一个超平面,分类器，寻找一个最优的超平面
#svm有很多核，这里使用线性核
#svm 核：line
#身高体重=》训练=》预测
import cv2
import numpy as np
import matplotlib.pyplot as plt
#1.准备data
rand1=np.array([[155,48],[159,50],[164,53],[168,56],[172,60]])
rand2=np.array([[152,53],[156,55],[160,56],[172,64],[176,65]])
#2.label
label=np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])
#3 .data
data=np.vstack((rand1,rand2))
data=np.array(data,dtype='float32')

#svm所有数据都要有label
#[155,48]--0 [152,53]--1
#监督学习 0负样本 1正样本

#4训练 SVM_sreate train predict
svm=cv2.ml.SVM_create()
#属性设置
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)

svm.train(data,cv2.ml.ROW_SAMPLE,label)

#预测
pt_data=np.vstack([[167,55],[162,57],[163,58],[166,56.55]])#0女生 1男生 输出为0110
pt_data=np.array(pt_data,dtype='float32')
print(pt_data)
pararr=svm.predict(pt_data)
def printRes(result):
    for i in range(len(result[1])):
        if result[1][i][0]==0:
            print('女生\n')
        else:
            print('男生\n')

printRes(pararr)
