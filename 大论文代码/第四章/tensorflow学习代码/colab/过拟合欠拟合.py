import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
x = np.random.uniform(-3, 3, size=100)  # 生成x特征 -3到3  100个
X = x.reshape(-1, 1)  # 将x编程100行1列的矩阵
y =   x ** 2 + x + 2 + np.random.normal(0, 0.1, size=100)  # 模拟的是标记y  对应的是x的二次函数
def line_regress(dim,name='None'):
    reg = PolynomialRegression(dim)
    reg.fit(X, y)
    reg.score(X, y)

    # 将预测值y_pre画图 对比真实y
    y_pre = reg.predict(X)
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_pre[np.argsort(x)], color='r')
    plt.title(label=name,fontsize=20)
    # plt.show()
    # 查看MSE
from sklearn.metrics import mean_squared_error

# mean_squared_error(y, y_pre)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
def PolynomialRegression(degree):
    return Pipeline([  # 构建Pipeline
        ("poly", PolynomialFeatures(degree=degree)),  # 构建PolynomialFeatures
        ("std_scaler", StandardScaler()),  # 构建归一化StandardScaler
        ("lin_reg", LinearRegression())  # 构建线性回归LinearRegression
    ])

def suit_fit():
    # 将Pipeline封装 方便使用


    # 设置degree=2 进行fit拟合
    poly2_reg = PolynomialRegression(2)
    poly2_reg.fit(X, y)

    # 求出MSE
    y2_pre = poly2_reg.predict(X)
    # mean_squared_error(y2_pre, y)

def over_fit():
    x = np.random.uniform(-3, 3, size=100)  # 生成x特征 -3到3  100个
    X = x.reshape(-1, 1)  # 将x编程100行1列的矩阵
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)  # 模拟的是标记y  对应的是x的二次函数

plt.subplot(1,3,1)
line_regress(1,'under_fit')
plt.subplot(1,3,2)
line_regress(3,'fit')
plt.subplot(1,3,3)
line_regress(20,'over_fit')
plt.show()