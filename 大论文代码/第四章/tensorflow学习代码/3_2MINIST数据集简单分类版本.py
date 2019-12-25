import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#每个批次的大小
batch_size=100
#计算一共多少批次
n_batch=mnist.train.num_examples//batch_size

TRAIN_STEPS=100000

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

node=400
W1=tf.Variable(tf.truncated_normal([784,node],stddev=0.1))
b1=tf.Variable(tf.truncated_normal([node],stddev=0.1))
layer1=tf.nn.relu(tf.matmul(x,W1)+b1)
W2=tf.Variable(tf.truncated_normal([node,10],stddev=0.1))
b2=tf.Variable(tf.truncated_normal([10],stddev=0.1))
prediction=tf.matmul(layer1,W2)+b2
# W=tf.Variable(tf.truncated_normal([784,10],stddev=0.1))
# b=tf.Variable(tf.constant(0.1,shape=[10]))
# prediction=tf.nn.relu(tf.matmul(x,W)+b)
prediction=tf.nn.softmax(prediction)
global_step=tf.Variable(0,trainable=False)

#可以衡量不同模型的优劣
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)
cross_entropy_mean=tf.reduce_mean(cross_entropy)
# loss=tf.reduce_mean(tf.square(y-prediction))
from tensorflow.contrib import layers
regularizer=layers.l2_regularizer(0.0001)
regularition=regularizer(W1)+regularizer(W2)

loss=cross_entropy_mean+regularition

#设置滑动平均模型
variable_average=tf.train.ExponentialMovingAverage(0.99,global_step)
variable_average_op=variable_average.apply(tf.trainable_variables())
#设置衰减的学习率
learning_rate=tf.train.exponential_decay(0.8,global_step,n_batch,0.99)
trainstep=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

init=tf.global_variables_initializer()

#返回true和false
correction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

#求准确率
accuracy=tf.reduce_mean(tf.cast(correction,tf.float32))

with tf.control_dependencies([trainstep,variable_average_op]):
    train_op=tf.no_op(name='train')

with tf.Session() as sess:
    sess.run(init)
    # sess.run(variable_average_op)

    for i in range(TRAIN_STEPS):
        xs, ys = mnist.train.next_batch(batch_size)
        sess.run(trainstep, feed_dict={x: xs, y: ys})
        if(i%1000==0):
            acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
            print('global setps is {0}'.format(str(sess.run(global_step))))
            print('learning rate is {0}'.format(str(sess.run(learning_rate))))
            print('after {0} trains,accuracy is {1}'.format(str(i),str(acc)))