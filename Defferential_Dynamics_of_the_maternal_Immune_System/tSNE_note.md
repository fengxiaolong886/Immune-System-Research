# 关于t-SNE降维方法

在[Differential Dynamics of the Maternal Immune System in Healthy Pregnancy and Preeclampsia](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6584811/# ) 这篇论文中用到了t-SNE的降维方法进行可视化。

论文原图是这样的：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-cefdf84d29efe803.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



# 1. 什么是t-SNE:

全名是t-distributed Stochastic Neighbor Embedding(t-SNE)，翻译过来应该可以叫学生t分布的随机邻点嵌入法。

**t-SNE将数据点之间的相似度转换为概率**。原始空间中的相似度由高斯联合概率表示，嵌入空间的相似度由“学生t分布”表示。t-SNE在一些降维方法中表现得比较好。因为**t-SNE主要是关注数据的局部结构**。

**通过原始空间和嵌入空间的联合概率的Kullback-Leibler（KL）散度来评估可视化效果的好坏，也就是说用有关KL散度的函数作为loss函数，然后通过梯度下降最小化loss函数，最终获得收敛结果**。

正式点来描述就是：

给定一组 $N$ 个点 $x_1, \cdots, x_N \in \mathbb{R}^d$, t-SNE 首先计算$x_i$ 和 $x_j$之间的相似度$p_{ij}$。这个相似度公式定义为：

$p_{ij} = (p_{i \mid j} + p_{j \mid i})/(2N)$

对于每个$i$都有$p_{j \mid i} \propto \exp(\|x_i-x_j\|^2/\sigma_i^2)$ ，这里就是用的高斯核了，只涉及到一个参数$\sigma_i$. 



for some parameter $\sigma_i$. Intuitively, the value $p_{ij}$ measures the `similarity' between points $x_i$ and $x_j$. t-SNE then aims to learn the lower dimensional points $y_1, \cdots, y_N \in \mathbb{R}^2$ such that if $q_{ij} \propto (1+\|y_i-y_j \|_2^2)^{-1}$, then $q_{ij}$ minimizes the Kullback–Leibler divergence of the distribution $\{q_{ij}\}$ from the distribution $\{p_{ij}\}$. For a more detailed explanation of the t-SNE algorithm, see \cite{orig_tsne}.

直观地讲，该值$p_{ij}$衡量点与点之间的相似性。 然后t-SNE学习较低的维度，以便在低维空间中将分布的KL散度最小化。

这里简单列一下t-SNE的算法：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-bb6f3026ff24e92b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

关于t-SNE的更多内容参考另外一篇论文[Visualizing Data using t-SNE](http://www.jmlr.org/papers/v9/vandermaaten08a.html ) .

# 2. t-SNE的问题

- t-SNE的计算复杂度很高，在数百万个样本数据集中可能需要几个小时，而PCA可以在几秒钟或几分钟内完成
- 只能限于二维或三维嵌入。
- 算法是随机的，具有不同种子的多次实验可以产生不同的结果。虽然选择loss最小的结果就行，但可能需要多次实验以选择超参数。

# 3. t-SNE的参数

这里列一下在TensorFlow中t-SNE相关的参数，其实参数很多，但是TensorFlow做了很多自动化处理，所以只考虑下面这几个：

* Dimension: 这个只是考虑输出结果是二维空间还是三维空间。

* Perplexity：这个可以叫困惑度，他说明了如何在数据的本地和全局方面之间取得平衡。也就是说通过困惑度去猜测某个点的邻居有多少个。这个对邻点数量的猜测可以对最后结果的图片复杂度影响很大。我们通过调整Perplexity可以分析出很多不同结果的降维图片。在上面的那篇原始论文（[Visualizing Data using t-SNE](http://www.jmlr.org/papers/v9/vandermaaten08a.html ) .）中提到：“**The performance of SNE is fairly robust to changes in the perplexity, and typical values are between 5 and 50**”， 也就是t-SNE的鲁棒性很不错，一般在5到50之间调整就可以了
* Learning Rate: 这个学习率是在运行t-SNE算法的时候，进行梯度更新的步长。学习率的设置可以自由调整，但是也要根据样本量的大小来调整，样本量少学习率就调小一点。
* Supervise: 这个参数是调整标签的重要程度的，我们输入的数据都是成对的（数据，标签），这个参数可以从0（不用标签）到10（全部标签都用到）进行调整。具体细节可以参考来自IBN的这篇文章[Interactive supervision with TensorBoard](https://www.ibm.com/blogs/research/2017/11/interactive-supervision-tensorboard/)

# 4. 实验工具:TensorBoard

TensorFlow是一个开源软件库，在机器学习，深度学习以及强化学习这些领域应用最广泛的最基础的工具，他好处之一是自动计算微分，然后很容易搭建神经网络并进行训练。

[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) 是用于可视化 TensorFlow 模型的训练过程的工具（the flow of tensors，在你安装 TensorFlow 的时候就已经安装了 TensorBoard。

他的组成比较复杂，功能比较多，参考这个结构：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-f6984e2c96ac3e1a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们主要只是使用EMBEDDINGS功能。



## 5. TensorFlow安装

* 安装Python 3.7.2
* pip安装tensorflow 1.13.1

# 6. 实验设定

这次实验我们只做做基础的使用MNIST观察下t-SNE的结果，只用了1000张手写数字图片。

MNIST是一个简单的计算机视觉数据集。它包含如下所示的手写数字的图片集:

![image.png](https://upload-images.jianshu.io/upload_images/15463866-3f50c095146d82ab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

MNIST数据保存在了[Yann LeCun的网站](https://github.com/argszero/tensorflow_cn/blob/master/tensorflow.org/tutorials/mnist/beginners),

下载的数据包含两部分，6万个训练数据(mnist.train)和10万各测试数据(mnist.test)。这个区分是非常重要的：在机器学习领域，我们分离出一部分数据，这部分数据我们不用来作训练，以此来保证我们学到的是通用的规则。

就像前面提到的那样，每个MNIST数据有都有两个部分：一个手写数字的图片，以及一个相应的标签。我们用"xs"表示图片，用"ys"表示标签。训练数据和测试数据都有xs和ys，以测试数据举例来说，训练图片是mnist.train.images，训练标签是mnist.train.labels。

每个图片都是28X28像素，我们可以将它理解为一个大数组。

![image.png](https://upload-images.jianshu.io/upload_images/15463866-78e4cd1459d1f1db.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这个数组可以flatten为一个包含28X28=784个数字的向量,如何flatten这个数组是没有关系的，因为我们在所有图片中都是相同的。从这个角度看，MNIST图片只是一堆784维空间中的一个点

我们将mnist看作是一个[60000,784]的张量(an n-dimensional array) 。第一个维度是图片的索引，第二个维度图片上像素的索引。这个张量的每个实体都是某个图片，某个像素的，像素亮度用0到1之间的数字表示

MNIST对应的标签是0～9的数字，描述了给定图片是哪个数字。基于本教程的目的，我们想要我们的标签作为"one-hot vectors",one-hot vector是一个大多数维度都是0，只有一个维度为1的向量。在我们这个例子里，第n个数字会表示为一个第n位为1的向量，比如0会表示为[1,0,0,0,0,01,0,0,0,0,0,0,0,0,0,0],相应的，mnist.train.labels是一个[60000,10]的float数组。

# 7. 实验的代码

我们实验的代码如下

这里我们主要使用的网络架构是这样的，输入图片进入第一层卷积神经网络，使用32个[5,5]形状的滤波器，并使用补偿为2的最大池化操作。然后进入第二层卷积神经网络，使用64个[5,5]形状的滤波器，并使用补偿为2的最大池化操作。最后把得到的张量铺平成一个一维度的向量，通过一个512个cell的全连接网络，最后再进入soft输出结果是数字1-10的概率。

使用cross-entropy作为损失函数，并使用优化算法

```
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
```



# 8. 实验结果

在上面的代码训练完成之后，我们把结果保存在了代码中定义的目录logs中。我们直接进入logs目录的父目录然后在CMD中启动tensorboard即可：

​     tensorboard --logdir logs    

启动过程如下所示：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-bc8a9e35fd9639b5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

然后我们通过chrome浏览器访问tensorboard，

http://localhost:6006

我们在PROJECTOR可以看到t-SNE的结果：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-0b8957514f1a1e21.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这个记过看起来这1000个图片在三维空间上确实是把不同的图片分开了。Perplexity,learning rate与supervise都随意调的，可以仔细调整看看。然后这里只跑了184次迭代，因为发现这里效果看起来不错，训练的迭代次数往后越多，这些堆分开得越远。效果很好，但是不好看。

我们看看在二维空间的可视化结果：

![image.png](https://upload-images.jianshu.io/upload_images/15463866-38f416d05b67354c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

同样的参数下也只训练了182个迭代，效果不错的。

此外最后我们再看看PCA降维的效果，感觉和T-SNE差别比较大，没有深入研究，可能TSNE是在动态训练，PCA这里只是静态的。注意PCA在三维上是需要手动选择特征的。

![image.png](https://upload-images.jianshu.io/upload_images/15463866-03813e56434fc2c8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

