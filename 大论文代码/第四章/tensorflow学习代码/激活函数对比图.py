# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 20:41:57 2018

@author: brucelau
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.array([0*item  if item<0 else item for item in x ])

x = np.linspace(-10,10,1000)
y_sigmoid = 1/(1+np.exp(-x))
y_tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
charactr_size=20
color='red'
row,col=2,3
fig = plt.figure()
# plot sigmoid
ax = fig.add_subplot(row,col,1)
ax.plot(x,y_sigmoid,color=color)
ax.grid()
ax.set_title('(a) Sigmoid ',fontsize=charactr_size)

ax = fig.add_subplot(row,col,1+3)
ax.plot(x,sigmoid(x)*(1-sigmoid(x)),color=color)
ax.grid()
ax.set_title('(d) Sigmoid 导数',fontsize=charactr_size)

# plot tanh
ax = fig.add_subplot(row,col,2)
ax.plot(x,y_tanh,color=color)
ax.grid()
ax.set_title('(b) Tanh',fontsize=charactr_size)

ax = fig.add_subplot(row,col,2+3)
ax.plot(x,1-tanh(x)**2,color=color)
ax.grid()
ax.set_title('(e) Tanh 导数',fontsize=charactr_size)

# plot relu
ax = fig.add_subplot(row,col,3)
y_relu = np.array([0*item  if item<0 else item for item in x ])
ax.plot(x,y_relu,color=color)
ax.grid()
ax.set_title('(c) ReLu',fontsize=charactr_size)

ax = fig.add_subplot(row,col,3+3)
y_relu = np.array([0*item  if item<0 else 1 for item in x ])
ax.plot(x,y_relu,color=color)
ax.grid()
ax.set_title('(f) ReLu 导数',fontsize=charactr_size)

# #plot leaky relu
# ax = fig.add_subplot(2,4,4)
# y_relu = np.array([0.2*item  if item<0 else item for item in x ])
# ax.plot(x,y_relu)
# ax.grid()
# ax.set_title('(d) Leaky ReLu')

# plt.tight_layout()
plt.show()