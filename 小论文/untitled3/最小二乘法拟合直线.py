import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-1, 1, 0.04)
c=np.random.normal(0,0.01,size=[3,3])
y =x**2+c

y1 = y + (np.random.rand(len(x)) - 0.5)

##################################
### 核心程序
# 使用函数y=ax^3+bx^2+cx+d对离散点进行拟合，最高次方需要便于修改，所以不能全部列举，需要使用循环
# A矩阵
m = []
for i in  range(7):  # 这里选的最高次为x^7的多项式
    a = x ** (i)
    m.append(a)
A = np.array(m).T
b = y1.reshape(y1.shape[0], 1)


##################################

def projection(A, b):
    AA = A.T.dot(A)  # A乘以A转置
    w = np.linalg.inv(AA).dot(A.T).dot(b)
    print(w)  # w=[[-0.03027851][ 0.1995869 ] [ 2.43887827] [ 1.28426472][-5.60888682] [-0.98754851][ 2.78427031]]
    return A.dot(w)


yw = projection(A, b)
yw.shape = (yw.shape[0],)

plt.plot(x, y, color='g', linestyle='-', marker='', label="Ideal curve")#理想曲线
plt.plot(x, y1, color='m', linestyle='', marker='o', label="konw data points")#已知数据点
plt.plot(x, yw, color='r', linestyle='', marker='.', label="fitting curve")#拟合曲线
plt.legend(loc='upper left')
plt.show()