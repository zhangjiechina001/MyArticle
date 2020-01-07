# -*- coding: utf-8 -*-
import pandas as pd
from numpy.random import shuffle
from sklearn import svm
import joblib
from sklearn import metrics
import cv2

input_file='data.csv'

def readData():
    """
    读取数据
    :return:
    """
    data = pd.read_csv(input_file, encoding='utf8')
    data = data.values
    shuffle(data)  # 随机打乱
    train_radio=0.7
    data_train = data[:int(train_radio * len(data)), :]  # 训练集矩阵
    data_test = data[int(train_radio * len(data)):, :]  # 测试集矩阵
    return data_train, data_test


def train(data_train, data_test,c):
    """
    训练
    :param data_train:
    :param data_test:
    :return:
    """
    x_train = data_train[:, 0:2500]   # 放大特征，矩阵2维
    y_train = data_train[:, -1].astype('str')
    x_test = data_test[:, 0:2500]   # 放大特征
    y_test = data_test[:, -1].astype('str')

    # C：错误项的惩罚系数。C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，也就是对测试数据的分类准确率降低。相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。对于训练样本带有噪声的情况，一般采用后者，把训练样本集中错误分类的样本作为噪声。
    # kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
    # kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
    # kernel='poly'时，多项式函数,degree 表示多项式的程度-----支持非线性分类。更高gamma值，将尝试精确匹配每一个训练数据集，可能会导致泛化误差和引起过度拟合问题。
    # kernel='sigmoid'时，支持非线性分类。更高gamma值，将尝试精确匹配每一个训练数据集，可能会导致泛化误差和引起过度拟合问题。
    # gamma：float参数 默认为auto。核函数系数，只对‘rbf’,‘poly’,‘sigmod’有效。如果gamma为auto，代表其值为样本特征数的倒数，即1/n_features.
    # decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
    # decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
    # probability：bool参数 默认为False，是否启用概率估计。 这必须在调用fit()之前启用，并且会fit()方法速度变慢。
    # cache_size：float参数 默认为200，指定训练所需要的内存，以MB为单位，默认为200MB。
    # class_weight：字典类型或者‘balance’字符串。默认为None，给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C.如果给定参数‘balance’，则使用y的值自动调整与输入数据中的类频率成反比的权重。
    # max_iter ：int参数 默认为-1，最大迭代次数，如果为-1，表示不限制
    model = svm.SVC(C=c, kernel='linear', gamma='auto', decision_function_shape='ovr', cache_size=500)

    model.fit(x_train, y_train)

    # result=model.predict([[6.5,2.8,4.6,1.5]])
    # print(result)
    joblib.dump(model, 'num_svm.model')
    return x_train, y_train, x_test, y_test

def eval(x_train, y_train, x_test, y_test):
    """
    评估
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """

    model = joblib.load('num_svm.model')
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print("训练集得分：" + str(train_score))
    print("测试集得分：" + str(test_score))

    return train_score,test_score



    # cm_train = metrics.confusion_matrix(y_train, model.predict(x_train))  # 训练集混淆矩阵
    # cm_test = metrics.confusion_matrix(y_test, model.predict(x_test))  # 测试集混淆矩阵
    # pd.DataFrame(cm_train, index=range(1, 6), columns=range(1, 6)).to_csv(outputfile1)
    # pd.DataFrame(cm_test, index=range(1, 6), columns=range(1, 6)).to_csv(outputfile2)
import numpy as np
i=0
def dettect_code(img):
    h,w=img.shape
    if(h<60):
        img=cv2.resize(img,(85*w//h,85))
    global i
    i=i+1
    ret, img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow(str(i),img)
    # _, img = cv2.threshold(img, 10, 1, cv2.THRESH_BINARY)
    #金字塔处理的位置很重要
    #小一点的尺寸一般为45*30
    #大一点的尺寸一般为80*20
    img = cv2.pyrDown(img)
    h,w=img.shape
    # size_ratio=h/50
    # newh=50
    # neww=50*w//h
    # img = cv2.resize(img,(neww,newh))

    tempplate=np.ones([50,50])
    # tempplate[:,:]=255
    tempplate[(50-h)//2:(50-h)//2+h,(50-w)//2:(50-w)//2+w]=img[:,:]


    model = joblib.load('E:\MyArticle\大论文代码\第四章\svm\\num_svm.model')
    result=model.predict(tempplate.reshape([1,50*50]))
    return result[0]

if __name__ == '__main__':
    data_train, data_test = readData()

    # x_train, y_train, x_test, y_test = train(data_train, data_test)
    # eval(x_train, y_train, x_test, y_test)

    nums=np.linspace(0.1,1,10)
    train_scores=[]
    test_scores=[]
    for num in nums:
        x_train, y_train, x_test, y_test = train(data_train, data_test,num)
        train_score,test_score=eval(x_train, y_train, x_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)

    import matplotlib.pyplot as plt
    plt.plot(nums,train_scores)
    plt.plot(nums, test_scores)
    plt.show()



