#导入相关的库
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import time
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt
import numpy as np
#这里用slim这个API来进行卷积网络构建
slim = tf.contrib.slim

#定义卷积神经网络模型
#网络架构是卷积网络--最大池化--卷积网络--最大池化---flatten---MLP-softmax的全连接MLP
def model(inputs, is_training, dropout_rate, num_classes, scope='Net'):
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm):
            net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='conv1')
            net = slim.max_pool2d(net, 2, stride=2, scope='maxpool1')
            tf.summary.histogram("conv1", net)

            net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='conv2')
            net = slim.max_pool2d(net, 2, stride=2, scope='maxpool2')
            tf.summary.histogram("conv2", net)

            net = slim.flatten(net, scope='flatten')
            fc1 = slim.fully_connected(net, 1024, scope='fc1')
            tf.summary.histogram("fc1", fc1)

            net = slim.dropout(fc1, dropout_rate, is_training=is_training, scope='fc1-dropout')
            net = slim.fully_connected(net, num_classes, scope='fc2')

            return net, fc1


def create_sprite_image(images):
    """更改图片的shape"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    sprite_image = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                sprite_image[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return sprite_image


def vector_to_matrix_mnist(mnist_digits):
    """把正常的mnist数字图片(batch,28*28)这个格式，转换为新的张量形状(batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))


def invert_grayscale(mnist_digits):
    """处理下图片颜色，黑色变白，白色边黑"""
    return 1 - mnist_digits


if __name__ == "__main__":
    # 定义参数
    #学习率
    learning_rate = 1e-4
    #定义迭代参数
    total_epoch = 500
    #定义批量
    batch_size = 200
    #程序运行中打印频率
    display_step = 20
    #程序运行中保存结果的频率
    save_step = 100
    load_checkpoint = False
    checkpoint_dir = "checkpoint"
    checkpoint_name = 'model.ckpt'
    #结果存放的路径
    logs_path = "logs"
    #定义我们使用多少个图片
    test_size = 1000
    #定义第二层路径
    projector_path = 'projector'

    # 网络参数
    n_input = 28 * 28   # 每个图片是28*28个像素，也就是784个特征
    n_classes = 10  # MNIST数据集有0-9是个类别的结果
    dropout_rate = 0.5  # Dropout的比率

    mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

    # 定义计算图
    x = tf.placeholder(tf.float32, [None, n_input], name='InputData')
    y = tf.placeholder(tf.float32, [None, n_classes], name='LabelData')
    is_training = tf.placeholder(tf.bool, name='IsTraining')
    keep_prob = dropout_rate

    logits, fc1 = model(x, is_training, keep_prob, n_classes)

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    tf.summary.scalar("loss", loss)

    with tf.name_scope('Accuracy'):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    projector_dir = os.path.join(logs_path, projector_path)
    path_metadata = os.path.join(projector_dir,'metadata.tsv')
    path_sprites = os.path.join(projector_dir, 'mnistdigits.png')
    # 检查结果目录的状态
    if not os.path.exists(projector_dir):
        os.makedirs(projector_dir)

    # 这里进行嵌入
    mnist_test = input_data.read_data_sets('MNIST-data', one_hot=False)
    batch_x_test = mnist_test.test.images[:test_size]
    batch_y_test = mnist_test.test.labels[:test_size]

    embedding_var = tf.Variable(tf.zeros([test_size, 1024]), name='embedding')
    assignment = embedding_var.assign(fc1)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = os.path.join(projector_path,'metadata.tsv')
    embedding.sprite.image_path = os.path.join(projector_path, 'mnistdigits.png')
    embedding.sprite.single_image_dim.extend([28,28])

    # 初始化变量
    init = tf.global_variables_initializer()

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()
    merged_summary_op = tf.summary.merge_all()

    # 运行计算图
    with tf.Session() as sess:
        sess.run(init)
        # Restore model weights from previously saved model
        prev_model = tf.train.get_checkpoint_state(checkpoint_dir)
        if load_checkpoint:
            if prev_model:
                saver.restore(sess, prev_model.model_checkpoint_path)
                print('Checkpoint found, {}'.format(prev_model))
            else:
                print('No checkpoint found')

        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        projector.visualize_embeddings(summary_writer, config)
        start_time = time.time()
        # 开始训练
        for epoch in range(total_epoch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # reshapeX = np.reshape(batch_x, [-1, 28, 28, 1])
            # 开始反向传播算法
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           is_training: True})
            if epoch % display_step == 0:
                # 计算损失和精度
                cost, acc, summary = sess.run([loss, accuracy, merged_summary_op],
                                              feed_dict={x: batch_x,
                                                         y: batch_y,
                                                         is_training: False})
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('epoch {}, training accuracy: {:.4f}, loss: {:.5f}, time: {}'
                      .format(epoch, acc, cost, elapsed_time))
                summary_writer.add_summary(summary, epoch)
            if epoch % save_step == 0:
                # 保存训练的结果
                sess.run(assignment, feed_dict={x: mnist.test.images[:test_size],
                                                y: mnist.test.labels[:test_size], is_training: False})
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                save_path = saver.save(sess, checkpoint_path)
                print("Model saved in file: {}".format(save_path))

        # 保存结果
        saver.save(sess, os.path.join(logs_path, "model.ckpt"), 1)
        # 创建可视化的图片
        to_visualise = batch_x_test
        to_visualise = vector_to_matrix_mnist(to_visualise)
        to_visualise = invert_grayscale(to_visualise)
        sprite_image = create_sprite_image(to_visualise)
        # 保存可视化的图片
        plt.imsave(path_sprites, sprite_image, cmap='gray')
        # 写文件
        with open(path_metadata, 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(batch_y_test):
                f.write("%d\t%d\n" % (index, label))

        print("训练完成")