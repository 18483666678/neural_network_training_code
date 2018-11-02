import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


class GNet:

    def __init__(self):
        with tf.variable_scope("gnet"):
            self.W1 = tf.Variable(tf.truncated_normal(shape=[110, 128], stddev=0.1))
            self.b1 = tf.Variable(tf.zeros([128]))

            self.W2 = tf.Variable(tf.truncated_normal(shape=[128, 784], stddev=0.1))

    def forward(self, x, label):
        x = tf.concat([x, label], 1)
        # print("gent concat>>", x.shape)
        y = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        y = tf.matmul(y, self.W2)

        return y

    def getParm(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope="gnet")


class DNet:

    def __init__(self):
        with tf.variable_scope("dnet"):
            self.W1 = tf.Variable(tf.truncated_normal(shape=[794, 128], stddev=0.1))
            self.b1 = tf.Variable(tf.zeros([128]))

            self.W2 = tf.Variable(tf.truncated_normal(shape=[128, 1], stddev=0.1))

    def forward(self, x, label):
        x = tf.concat([x, label], 1)
        # print("dnet concat>>", x.shape)
        y = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        y = tf.nn.sigmoid(tf.matmul(y, self.W2))

        return y

    def getParam(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope="dnet")


class MyGANNet:

    def __init__(self):
        self.real_x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.true_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.gen_x = tf.placeholder(dtype=tf.float32, shape=[None, 100])
        self.false_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        self.gnet = GNet()
        self.dnet = DNet()

    def forward(self):
        self.real_d_out = self.dnet.forward(self.real_x, self.label)

        self.gen_out = self.gnet.forward(self.gen_x, self.label)
        self.gen_d_out = self.dnet.forward(self.gen_out, self.label)

    def backward(self):
        self.d_loss = tf.reduce_mean((self.real_d_out - self.true_y) ** 2) \
                      + tf.reduce_mean((self.gen_d_out - self.false_y) ** 2)
        self.d_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.d_loss,
                                                                   var_list=self.dnet.getParam())

        self.gen_loss = tf.reduce_mean((self.gen_d_out - self.true_y) ** 2)
        self.gen_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.gen_loss,
                                                                     var_list=self.gnet.getParm())


if __name__ == '__main__':
    net = MyGANNet()
    net.forward()
    net.backward()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        plt.ion()
        for epoch in range(100000):
            xs, ys = mnist.train.next_batch(128)
            true_ys = np.ones([128, 1])

            gen_xs = np.random.uniform(-1, 1, (128, 100))
            gen_ys = np.zeros([128, 1])
            d_loss, _ = sess.run([net.d_loss, net.d_optimizer],
                                 feed_dict={net.real_x: xs, net.true_y: true_ys, net.gen_x: gen_xs,
                                            net.false_y: gen_ys, net.label: ys})

            gen_xs_true = np.random.uniform(-1, 1, (128, 100))
            gen_ys_true = np.ones([128, 1])
            gen_out, g_loss_t, _ = sess.run([net.gen_out, net.gen_loss, net.gen_optimizer],
                                            feed_dict={net.gen_x: gen_xs_true, net.true_y: gen_ys_true, net.label: ys})

            if epoch % 100 == 0:
                print("d_loss: {0}, g_loss_t: {1}".format(d_loss, g_loss_t))
                test_gen_xs = np.random.uniform(-1, 1, (1, 100))

                """ random condition, random noise """
                y = np.random.choice(10, 1)
                y_one_hot = np.zeros((1, 10))
                y_one_hot[np.arange(1), y] = 1

                test_gen_out = sess.run([net.gen_out], feed_dict={net.gen_x: test_gen_xs, net.label: y_one_hot})
                test_gen_out = np.reshape(test_gen_out[0], [28, 28])
                plt.clf()
                plt.subplot(221)
                plt.imshow(test_gen_out)
                print(test_gen_out)
                plt.subplot(222)
                plt.imshow(xs[0].reshape([28, 28]))
                plt.subplot(223)
                plt.imshow(gen_out[0].reshape([28, 28]))
                # print(type(gen_out), np.shape(gen_out))
                plt.pause(0.5)
        plt.ioff()
