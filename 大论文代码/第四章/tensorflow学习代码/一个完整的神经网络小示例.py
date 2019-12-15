import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from numpy.random import RandomState
import tensorflow.compat.v1 as tfv

#定义训练集数据batch的大小
batch_size=8

#定义神经网络参数
w1=tfv.Variable(tfv.random.normal([2,3],stddev=1,seed=1))
w2=tfv.Variable(tfv.random.normal([3,1],stddev=1,seed=1))

x=tfv.placeholder(tfv.float32,shape=(None,2),name='x_input')
y_=tfv.placeholder(tfv.float32,shape=(None,1),name='y_input')

a=tfv.matmul(x,w1)
y=tfv.matmul(a,w2)

#定义损失函数和反向传播算法
y=tfv.sigmoid(y)
cross_entropy=-tfv.reduce_mean(y_*tfv.log(tfv.clip_by_value(y,1e-10,1.0))+(1-y_)*tfv.log(tfv.clip_by_value(1-y,1e-10,1.0)))
train_step=tfv.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm=RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2)
Y=[[int(x1+x2<1)] for (x1,x2) in X]

with tfv.Session() as sess:
    init_op=tfv.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))

    STEPS=5000
    for i in range(STEPS):
        start=(i*batch_size)%dataset_size
        end=min(start+batch_size,dataset_size)

        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

        if(i%1000==0):
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print('After {0} train steps,crosee entropy on all data is {1}'.format(str(i),str(total_cross_entropy)))

    print(sess.run(w1))
    print(sess.run(w2))