import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
#用numpy生成200个点
x_data=np.linspace(-0.5,0.5,2000)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

#定义两个placeholder
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#定义网络中间层
weight_l1=tf.Variable(tf.random.normal([1,10]))
biases_l1=tf.Variable(tf.zeros([1,10]))
w_l1=tf.matmul(x,weight_l1)+biases_l1
L1=tf.nn.tanh(w_l1)

#定义输出层
weight_l2=tf.Variable(tf.random_normal([10,1]))
biases_l2=tf.Variable(tf.zeros([1,1]))

w_l2=tf.matmul(L1,weight_l2)+biases_l2
prediction=tf.nn.tanh(w_l2)

#二次代价函数
loss=tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
        if(step%100==0):
            print(step,sess.run([weight_l1,weight_l2]))
        #获得预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-')
    plt.show()