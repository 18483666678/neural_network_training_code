import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


class EncodeNet:

    def __init__(self):
        self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1, dtype=tf.float32))
        self.conv1_b = tf.Variable(tf.zeros([16]))

        self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1, dtype=tf.float32))
        self.conv2_b = tf.Variable(tf.zeros([32]))

    def forward(self, x):
        conv1 = tf.nn.relu(tf.nn.conv2d(x, self.conv1_w, [1, 1, 1, 1], "SAME") + self.conv1_b)
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, self.conv2_w, [1, 1, 1, 1], "SAME") + self.conv2_b)
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        return pool2


class DecodeNet:

    def __init__(self):
        self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1, dtype=tf.float32))
        self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1, dtype=tf.float32))

    def forward(self, x):
        deconv1 = tf.nn.conv2d_transpose(x, self.conv1_w, output_shape=[128, 14, 14, 16], strides=[1, 2, 2, 1],
                                         padding='SAME')
        deconv2 = tf.nn.conv2d_transpose(deconv1, self.conv2_w, output_shape=[128, 28, 28, 1], strides=[1, 2, 2, 1],
                                         padding='SAME')

        return deconv2


class MyNet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))

        self.encode = EncodeNet()
        self.decode = DecodeNet()

    def forword(self):
        y = self.encode.forward(self.x)
        self.out = self.decode.forward(y)

    def backword(self):
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(self.out, [-1, 28 * 28 * 1]),
                                                    labels=tf.reshape(self.x, [-1, 28 * 28 * 1])))
        tf.summary.scalar("loss", self.loss)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)


if __name__ == '__main__':

    net = MyNet()
    net.forword()
    net.backword()
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    plt.ion()
    init = tf.global_variables_initializer()
    merge = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("./logs", sess.graph)
        for i in range(10000):
            xs, ys = mnist.train.next_batch(128)
            batch_xs = np.reshape(xs, [128, 28, 28, 1])
            summary, out, loss, _ = sess.run([merge, net.out, net.loss, net.opt], feed_dict={net.x: batch_xs})

            writer.add_summary(summary, i)
            if i % 100 == 0:
                print("loss: {0}".format(loss))
                for pic in range(10):
                    a[0][pic].imshow(np.reshape(batch_xs[pic], [28, 28]))
                    a[1][pic].imshow(np.reshape(out[pic], [28, 28]))
                plt.pause(1)
