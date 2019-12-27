# -*- coding:utf-8 -*-
import cv2
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    def weight_variable(shape):
        # 产生随机变量
        # truncated_normal：选取位于正态分布均值=0.1附近的随机值
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def conv2d(x, W):
        # stride = [1,水平移动步长,竖直移动步长,1]
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool_2x2(x):
        # stride = [1,水平移动步长,竖直移动步长,1]
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    # 读取MNIST数据集
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    sess = tf.InteractiveSession()

    # 预定义输入值X、输出真实值Y    placeholder为占位符
    x = tf.placeholder(tf.float32, shape=[None, 784],name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, 10],name='y-input')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # print(x_image.shape)  #[n_samples,28,28,1]

    # 卷积层1网络结构定义
    # 卷积核1：patch=5×5;in size 1;out size 32;激活函数reLU非线性处理
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28*28*32
    h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14*32#卷积层2网络结构定义

    # 卷积核2：patch=5×5;in size 32;out size 64;激活函数reLU非线性处理
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14*14*64
    h_pool2 = max_pool_2x2(h_conv2)  # output size 7 *7 *64

    # 全连接层1
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # [n_samples,7,7,64]->>[n_samples,7*7*64]
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 减少计算量dropout

    # prediction=tf.Variable()
    # 全连接层2
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_output=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # prediction = tf.Variable(tf.constant(0.0,shape=[None,10],dtype=tf.float32),name='prediction')
    prediction = tf.nn.softmax(y_output)

    # 二次代价函数:预测值与真实值的误差
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=prediction))

    # 梯度下降法:数据太庞大,选用AdamOptimizer优化器
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # 结果存放在一个布尔型列表中
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()  # defaults to saving all variables
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step", i, "training accuracy", train_accuracy)

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # 保存模型参数
    saver.save(sess, 'model/model.ckpt')

def detect(img):
    # img=img[10:90,20:80]
    img = 255 - img[:, :]
    _,tempimg=cv2.threshold(img,100,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(tempimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        area=cv2.contourArea(contours[i])
        if(area>100):
            cutNum = 3
            x, y, w, h = cv2.boundingRect(contours[i])
            img = img[y - cutNum:y + h + cutNum, x - cutNum:x + w + cutNum]
            break

    img=cv2.resize(img,(28,28))

    # kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    # img=cv2.dilate(img,kernel)
    img=(img[:,:]/255).astype('float32')
    _, img = cv2.threshold(img, 0.2, 1, cv2.THRESH_TOZERO)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey()

    img=img.reshape(1,28*28)

    with tf.Session() as sess:
        # 载入meta文件
        saver = tf.train.import_meta_graph(r'E:\大论文\大论文代码\第四章\tensorflow学习代码\卷积神经网络\model\model.ckpt.meta')
        # 载入最近的一次保存的模型文件
        saver.restore(sess, tf.train.latest_checkpoint(r"E:\大论文\大论文代码\第四章\tensorflow学习代码\卷积神经网络\model/"))
        # 建立图
        graph = tf.get_default_graph()
        # 初始化所有的变量
        sess.run(tf.global_variables_initializer())
        # 获取网络中的变量
        X = graph.get_tensor_by_name('x-input:0')
        y=graph.get_tensor_by_name('y-input:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        result = graph.get_tensor_by_name('add_3:0')
        # 这上面的‘data’、‘keep_prob’、‘prediction’都是我们在训练时定义的变量名称
        # prediction=sess.run(result, feed_dict={X: mnist.test.images[0:30]})
        prediction = sess.run(result, feed_dict={X: img,keep_prob:0.5})
        # print(sess.run(tf.argmax(mnist.test.labels[0:30],1)))
        print(sess.run(tf.argmax(prediction, 1)))


def main(argv=None):
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # train(mnist)
    img = cv2.imread('formatedImg\\9.jpg', cv2.IMREAD_GRAYSCALE)
    detect(img)

if __name__ == '__main__':
    main()
    # train()