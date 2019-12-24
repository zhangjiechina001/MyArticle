import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#每个批次的大小
batch_soze=100
#计算一共多少批次
n_batch=mnist.train.num_examples//batch_soze

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
prediction=tf.matmul(x,W)+b
prediction=tf.nn.softmax(prediction)

loss=tf.reduce_mean(tf.square(y-prediction))

trainstep=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()

#返回true和false
correction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

#求准确率
accuracy=tf.reduce_mean(tf.cast(correction,tf.float32))


