import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class GNet:

    def __init__(self):
        with tf.variable_scope("gnet"):
            self.w1 = tf.Variable(tf.truncated_normal(shape=[128, 1024], stddev=0.1))
            self.b1 = tf.Variable(tf.zeros([1024]))

            self.w2 = tf.Variable(tf.truncated_normal(shape=[1024, 32 * 7 * 7], stddev=0.1))
            self.b2 = tf.Variable(tf.zeros([32 * 7 * 7]))
            self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1, dtype=tf.float32))
            self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1, dtype=tf.float32))

    def forward(self, x):
        y = tf.nn.leaky_relu(tf.matmul(x, self.w1) + self.b1)
        y = tf.nn.leaky_relu(tf.matmul(y, self.w2) + self.b2)

        y = tf.reshape(y, [-1, 7, 7, 32])
        deconv1 = tf.nn.conv2d_transpose(y, self.conv1_w, output_shape=[100, 14, 14, 16], strides=[1, 2, 2, 1],
                                         padding='SAME')
        deconv2 = tf.nn.conv2d_transpose(deconv1, self.conv2_w, output_shape=[100, 28, 28, 1], strides=[1, 2, 2, 1],
                                         padding='SAME')

        return deconv2

    def getParam(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope="gnet")


class DNet:

    def __init__(self):
        with tf.variable_scope("dnet"):
            self.conv1_w = tf.Variable(
                tf.truncated_normal([3, 3, 1, 16], dtype=tf.float32, stddev=0.1))
            self.conv1_b = tf.Variable(tf.zeros([16]))

            self.conv2_w = tf.Variable(
                tf.truncated_normal([3, 3, 16, 32], dtype=tf.float32, stddev=0.1))
            self.conv2_b = tf.Variable(tf.zeros([32]))

            self.w1 = tf.Variable(tf.truncated_normal([7 * 7 * 32, 128], stddev=0.1))
            self.b1 = tf.Variable(tf.zeros([128]))

            self.w2 = tf.Variable(tf.truncated_normal([128, 1], stddev=0.1))

    def forward(self, x):
        conv1 = tf.nn.relu(tf.nn.conv2d(x, self.conv1_w, strides=[1, 1, 1, 1],
                                        padding='SAME') + self.conv1_b)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

        conv2 = tf.nn.relu(
            tf.nn.conv2d(pool1, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b)
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        flat = tf.reshape(pool2, [-1, 7 * 7 * 32])

        y = tf.matmul(flat, self.w1) + self.b1
        tf.summary.histogram("Discriminator pure output", y)
        y0 = tf.nn.relu(y)
        y1 = tf.nn.dropout(y0, keep_prob=0.9)
        y_out = tf.matmul(y1, self.w2)

        return y_out

    def getParam(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope="dnet")


class Net:

    def __init__(self):
        self.r_x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        self.t_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.g_x = tf.placeholder(dtype=tf.float32, shape=[None, 128])
        self.f_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.gnet = GNet()
        self.dnet = DNet()

        self.forward()
        self.backward()

    def forward(self):
        self.r_d_out = self.dnet.forward(self.r_x)

        self.g_out = self.gnet.forward(self.g_x)
        print("g_out shape: ", self.g_out.shape)
        self.g_d_out = self.dnet.forward(self.g_out)

    def backward(self):
        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.r_d_out, labels=self.t_y)) \
                      + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_d_out, labels=self.f_y))

        tf.summary.scalar("Discriminator loss", self.d_loss)
        self.d_opt = tf.train.AdamOptimizer(0.0001).minimize(self.d_loss, var_list=self.dnet.getParam())

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_d_out, labels=self.t_y))
        tf.summary.scalar("Generator loss", self.g_loss)
        self.g_opt = tf.train.AdamOptimizer(0.0001).minimize(self.g_loss, var_list=self.gnet.getParam())


if __name__ == '__main__':
    net = Net()
    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("./logs", sess.graph)

        plt.ion()
        for epoch in range(10000000):
            t_xs, _ = mnist.train.next_batch(100)
            t_xs = t_xs.reshape([100, 28, 28, 1])
            t_ys = np.ones(shape=[100, 1])

            f_xs = np.random.uniform(-1, 1, (100, 128))
            f_ys = np.zeros(shape=[100, 1])

            if epoch % 3 == 0:
                summary, _d_loss, _ = sess.run([merged, net.d_loss, net.d_opt],
                                               feed_dict={net.r_x: t_xs, net.t_y: t_ys, net.g_x: f_xs, net.f_y: f_ys})

            imgs, _g_loss, _ = sess.run([net.g_out, net.g_loss, net.g_opt],
                                        feed_dict={net.g_x: f_xs, net.t_y: t_ys})

            writer.add_summary(summary, epoch)

            if epoch % 100 == 0:
                print("epoch: {}, d_loss: {}, g_loss: {}".format(epoch, _d_loss, _g_loss))
                # test_xs = np.random.uniform(-1, 1, (100, 128))
                # imgs = sess.run(net.g_out, feed_dict={net.g_x: test_xs})
                img = np.reshape(imgs[0], (28, 28))
                # print(img)
                plt.clf()
                plt.imshow(img)
                plt.pause(1)
        plt.ioff()
