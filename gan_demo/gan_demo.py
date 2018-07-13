import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class GNet:

    def __init__(self):
        with tf.variable_scope("gnet"):
            self.W1 = tf.Variable(tf.truncated_normal(shape=[100, 128], stddev=0.1))
            self.b1 = tf.Variable(tf.zeros([128]))

            self.W2 = tf.Variable(tf.truncated_normal(shape=[128, 784], stddev=0.1))

    def forward(self, x):
        y = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        y = tf.matmul(y, self.W2)

        return y

    def getParm(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope="gnet")

class DNet:

    def __init__(self):
        with tf.variable_scope("dnet"):
            self.W1 = tf.Variable(tf.truncated_normal(shape=[784, 128], stddev=0.1))
            self.b1 = tf.Variable(tf.zeros([128]))

            self.W2 = tf.Variable(tf.truncated_normal(shape=[128, 1], stddev=0.1))

    def forward(self, x):
        y = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        y = tf.nn.sigmoid(tf.matmul(y, self.W2))

        return y

    def getParam(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope="dnet")


class MyGANNet:

    def __init__(self):
        self.real_x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        self.true_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.gen_x = tf.placeholder(dtype=tf.float32, shape=[None, 100])
        self.false_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.gnet = GNet()
        self.dnet = DNet()

    def forward(self):
        self.real_d_out = self.dnet.forward(self.real_x)

        self.gen_out = self.gnet.forward(self.gen_x)
        self.gen_out = tf.reshape(self.gen_out, shape=[-1, 28, 28, 1])
        self.gen_d_out = tf.dnet.forward(self.gen_out)

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