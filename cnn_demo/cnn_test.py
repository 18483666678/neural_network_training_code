import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


class CNNNet:

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])  # 这里因为要用卷积核，所以要依据tensorflow的N H W C，因为灰度c为1
        self.y = tf.placeholder(tf.float32, [None, 10])  # 多分类的方法，一共有10个结果

        self.dp = tf.placeholder(tf.float32)

        self.conv1_w = tf.Variable(
            tf.random_normal([3, 3, 1, 16], dtype=tf.float32, stddev=0.1))  # 3成3的卷积核，1个灰度，16个特征（超参数）#标准差弄小点
        self.conv1_b = tf.Variable(tf.zeros([16]))  # 把第一次的偏置设为0
        tf.summary.histogram("conv1_w", self.conv1_w)

        self.conv2_w = tf.Variable(tf.random_normal([3, 3, 16, 32], dtype=tf.float32, stddev=0.1))  # 其中前面是卷积核，16和32是超参数
        self.conv2_b = tf.Variable(tf.zeros([32]))  # 把第一次的偏置设为0
        tf.summary.histogram("conv2_w", self.conv2_w)

        self.w1 = tf.Variable(tf.random_normal([7 * 7 * 32, 128], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([128]))
        tf.summary.histogram("mlp_w1", self.w1)

        self.w2 = tf.Variable(tf.random_normal([128, 10], stddev=0.1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros([10]))
        tf.summary.histogram("mlp_w2", self.w2)

    def forward(self):
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x, self.conv1_w, strides=[1, 1, 1, 1],
                                             padding='SAME') + self.conv1_b)  # 第一个：batch上移动1代表不错过任何批次，内部两个1，是x轴步长，和y轴步长和通道
        self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME')  # 14*14 池化1，2，2，1，在其中2*2是池化形状，，后面1，2，2，1中的2*2是步长

        self.conv2 = tf.nn.relu(
            tf.nn.conv2d(self.pool1, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b)
        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 7*7

        self.flat = tf.reshape(self.pool2, [-1, 7 * 7 * 32])  # 其中-1是剩下所有的，其它是图片的大小和通道（从新整理图像）

        self._y0 = tf.matmul(self.flat, self.w1) + self.b1
        tf.summary.histogram("mlp_pure_out", self._y0)
        self.y0 = tf.nn.relu(self._y0)
        self.y1 = tf.nn.dropout(self.y0, keep_prob=self.dp)
        self.y_out = tf.nn.softmax(tf.matmul(self.y1, self.w2) + self.b2)

    def backward(self):
        # cross_entropy =tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)#这个是误差方程
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_out, labels=self.y))
        tf.summary.scalar("loss_cn", self.cross_entropy)
        # optimizer = tf.train.AdamOptimizer(0.011)#这个是op一个操作，梯度下降方法，学习率为0.5
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.y_out, 1), tf.argmax(self.y, 1))
        self.rst = tf.cast(self.correct_prediction, "float")
        self.accuracy = tf.reduce_mean(self.rst)


if __name__ == '__main__':
    # 调用类
    net = CNNNet()
    # 调用类的方法。
    net.forward()
    net.backward()

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("./logs", sess.graph)

        # 显示一张相片
        # saver.restore(sess,"save/xx.dpk")
        # plt.ion()
        # for epoch in range(1000):
        #     batch_xs, batch_ys = mnist.train.next_batch(1)
        #     # batch_xs = batch_xs.reshape([28, 28])
        #     batch_xs = batch_xs.reshape([1, 28, 28, 1])  # 这个地方要注意，传入参数满足N,H,W,C的传参标准
        #     _, loss, acc, pool2 = sess.run([net.optimizer, net.cross_entropy, net.accuracy, net.pool2],
        #                             feed_dict={net.x: batch_xs, net.y: batch_ys, net.dp: 0.9})
        #     img_p2 = np.reshape(pool2[:, :, :, 0], [7, 7])
        #     print(np.shape(img_p2))
        #     # print(img_p2)
        #     plt.imshow(img_p2)
        #     plt.pause(1)
        # plt.ioff()


        import time
        cc = time.time()
        for i in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(128)
            batch_xs = batch_xs.reshape([128, 28, 28, 1])  # 这个地方要注意，传入参数满足N,H,W,C的传参标准
            summary, _, loss, acc = sess.run([merged, net.optimizer, net.cross_entropy, net.accuracy],
                                    feed_dict={net.x: batch_xs, net.y: batch_ys, net.dp: 0.9})
            writer.add_summary(summary, i)
            # if i%100 == 0:
            #     saver.save(sess,"save/xx.dpk")

            # print(loss)
            if i % 100 == 0:
                print("精度：{0}".format(acc))
            # print(type(loss))
        print(time.time() - cc)
