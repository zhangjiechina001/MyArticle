import tensorflow as tf
import numpy as np

#使用numpy生成一百个随机点
x_data=np.random.rand(100)
y_data=0.1*x_data**2+0.9*x_data+0.7

#构造一个现象模型
b=tf.Variable(0.)
k=tf.Variable(0.)
k1=tf.Variable(0.)
y=k1*x_data**2+k*x_data+b

# init=tf.Variable.initializer()
#定义二次代价函数
loss=tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法来进行训练的优化器
optimizer=tf.train.GradientDescentOptimizer(0.2)
#定义一个最小化代价函数
train=optimizer.minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(20000):
        sess.run(train)

        if step%20==0:
            print(step,sess.run([k1,k,b]))
