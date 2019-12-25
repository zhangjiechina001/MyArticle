import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data

#MNIST相关常数
INPUT_NODE=784
OUTPUT_NODE=10

#配置神经网络参数
LAYER1_NODE=400

BATCH_SIZE=100

LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99

REGULARIZATION_RATE=0.0001

TRAIN_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

def inference(input_tensor,avg_class,w1,b1,w2,b2):
    #输出层使用不使用relu ：0.9816     使用relu:0.9818
    if avg_class==None:
        layers1=tf.nn.relu(tf.matmul(input_tensor,w1)+b1)
        return tf.nn.relu(tf.matmul(layers1,w2)+b2)
    else:
        layers1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(w1))+avg_class.average(b1))
        return tf.nn.relu(tf.matmul(layers1,avg_class.average(w2))+avg_class.average(b2))

def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    w1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    b1=tf.Variable(tf.truncated_normal([LAYER1_NODE],stddev=0.1))

    w2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    b2=tf.Variable(tf.truncated_normal([OUTPUT_NODE],stddev=0.1))

    prediction=inference(x,None,w1,b1,w2,b2)
    prediction=tf.nn.softmax(prediction)

    global_step=tf.Variable(0,trainable=False)

    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    average_y=inference(x,variable_averages,w1,b1,w2,b2)

    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    from tensorflow.contrib import layers
    regularizer=layers.l2_regularizer(REGULARIZATION_RATE)
    regularization=regularizer(w1)+regularizer(w2)
    loss=cross_entropy_mean+regularization

    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')

    correction_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

    accuracy=tf.reduce_mean(tf.cast(correction_prediction,tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        validate_feed={x:mnist.validation.images,
                       y_:mnist.validation.labels}

        test_feed={x:mnist.test.images,y_:mnist.test.labels}

        for i in range(TRAIN_STEPS):
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print('after {0} trains,validation accuracy use average model is {1}'.format(str(i),str(validate_acc)))
                print('learning_rate:{0}'.format(str(sess.run(learning_rate))))

            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print('teain end! after {0} trains,validation accuracy use average model is {1}'.format(str(TRAIN_STEPS),str(test_acc)))

def main(argv=None):
    mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
    train(mnist)

if __name__=='__main__':
    main()