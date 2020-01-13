import tensorflow as tf


with tf.Session() as sess:
    #载入meta文件
    saver = tf.train.import_meta_graph('./event_2classfication/module/model-99999.meta')
    #载入最近的一次保存的模型文件
    try:
        saver.restore(sess, tf.train.latest_checkpoint("./event_2classfication/module/"))
    except tf.errors.NotFoundError as e:
        print("Can't load checkpoint")
        print("%s" % str(e))

    # saver.restore(sess, tf.train.latest_checkpoint("E:\MyArticle\大论文代码\第四章\\tensorflow学习代码\colab\event_2classfication\module/"))
    #建立图
    graph = tf.get_default_graph()
    #初始化所有的变量
    sess.run(tf.global_variables_initializer())
    #获取网络中的变量
    X = graph.get_tensor_by_name("input/x_input:0")
    keep_prob = graph.get_tensor_by_name("fc_1/keep_prob/keep_prob:0")
    result = graph.get_tensor_by_name("fc_3/add:0")
    import cv2

    img = cv2.imread('getImages_format/1.jpg', cv2.IMREAD_GRAYSCALE)
    img = (img[:, :] / 255).astype('float32')
    temp = img.reshape([1, 28 * 28])
    #这上面的‘data’、‘keep_prob’、‘prediction’都是我们在训练时定义的变量名称
    print(sess.run(result, feed_dict={X: temp, keep_prob: 1.0}))
