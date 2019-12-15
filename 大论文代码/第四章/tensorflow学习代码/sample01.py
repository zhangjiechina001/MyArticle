import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
# a=tf.constant([3.0,5.0])
# b=tf.constant([8.0,10.0])
# result=a+b
# sess=tf.Session()
# variable=tf.constant(tf.zeros([2,3]))
# print(a.graph is tf.compat.v1.get_default_graph())
# with tf.compat.v1.Session as sess:
#     print(sess.run(a))
    # print(result)

w1=tf.Variable(tf.compat.v1.random.normal((2,3),stddev=1,seed=1))
w2=tf.Variable(tf.compat.v1.random.normal((3,1),stddev=1,seed=1))

x=tf.compat.v1.placeholder(tf.float32,shape=(2,2),name='input_data')
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
initer=tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(initer)
    # sess.run(w2.initializer)
    print(sess.run(y,feed_dict={x:[[0.7,0.99],[0.8,0.5]]}))
