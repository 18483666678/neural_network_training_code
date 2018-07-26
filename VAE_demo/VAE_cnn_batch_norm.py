# Base on VAE_cnn.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


def bn(x, is_training):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training)


class EncoderNet:

    def __init__(self):
        self.conv1_w = tf.Variable(
            tf.truncated_normal([3, 3, 1, 16], dtype=tf.float32, stddev=0.1))
        self.conv1_b = tf.Variable(tf.zeros([16]))

        self.conv2_w = tf.Variable(
            tf.truncated_normal([3, 3, 16, 32], dtype=tf.float32, stddev=0.1))
        self.conv2_b = tf.Variable(tf.zeros([32]))

        self.w1 = tf.Variable(tf.truncated_normal([7 * 7 * 32, 512], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([512]))

        self.logvar_w = tf.Variable(tf.truncated_normal(shape=[512, 128], stddev=0.1))
        self.mean_w = tf.Variable(tf.truncated_normal(shape=[512, 128], stddev=0.1))

    def forward(self, x, is_training):
        x = tf.reshape(x, [-1, 28, 28, 1])
        conv1 = tf.nn.leaky_relu(bn(tf.nn.conv2d(x, self.conv1_w, strides=[1, 2, 2, 1], padding='SAME') + self.conv1_b,
                                    is_training=is_training))
        # conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #                        padding='SAME')

        conv2 = tf.nn.leaky_relu(
            bn(tf.nn.conv2d(conv1, self.conv2_w, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_b,
               is_training=is_training))
        # conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        flat = tf.reshape(conv2, [-1, 7 * 7 * 32])
        y = tf.nn.relu(bn(tf.matmul(flat, self.w1) + self.b1, is_training=is_training))

        logVar = tf.matmul(y, self.logvar_w)
        mean = tf.matmul(y, self.mean_w)

        return mean, logVar


class DecoderNet:

    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal(shape=[128, 1024], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([1024]))

        self.w2 = tf.Variable(tf.truncated_normal(shape=[1024, 32 * 7 * 7], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([32 * 7 * 7]))
        self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1, dtype=tf.float32))
        self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1, dtype=tf.float32))

    def forward(self, x, is_training):
        y = tf.nn.leaky_relu(bn(tf.matmul(x, self.w1) + self.b1, is_training=is_training))
        y = tf.nn.leaky_relu(bn(tf.matmul(y, self.w2) + self.b2, is_training=is_training))

        y = tf.reshape(y, [-1, 7, 7, 32])
        deconv1 = tf.nn.leaky_relu(
            bn(tf.nn.conv2d_transpose(y, self.conv1_w, output_shape=[100, 14, 14, 16], strides=[1, 2, 2, 1],
                                      padding='SAME'), is_training=is_training))
        deconv2 = tf.nn.conv2d_transpose(deconv1, self.conv2_w, output_shape=[100, 28, 28, 1], strides=[1, 2, 2, 1],
                                         padding='SAME')

        y = tf.reshape(deconv2, [-1, 784])
        return y


class VAENet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

        self.learning_rate = 0.0002
        self.beta1 = 0.5

        self.encoder = EncoderNet()
        self.decoder = DecoderNet()

    def forward(self):
        self.mean, self.logVar = self.encoder.forward(self.x, is_training=True)

        noise = tf.random_normal(shape=[128])
        self.std = tf.sqrt(tf.exp(self.logVar))
        y = self.std * noise + self.mean
        self.out = self.decoder.forward(y, is_training=True)

    def backward(self):
        out_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.x))
        kl_loss = tf.reduce_mean(0.5 * (-self.logVar + self.mean ** 2 + tf.exp(self.logVar) - 1))
        self.loss = out_loss + kl_loss

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.loss)


if __name__ == '__main__':
    net = VAENet()
    net.forward()
    net.backward()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        plt.ion()
        for epoch in range(1000000):
            xs, ys = mnist.train.next_batch(100)
            out, loss, _ = sess.run([net.out, net.loss, net.opt], feed_dict={net.x: xs})

            if epoch % 100 == 0:
                print("epoch: {}, loss: {}".format(epoch, loss))
                img = np.reshape(out[0], [28, 28])
                plt.clf()
                plt.imshow(img)
                plt.pause(0.5)
        plt.ioff()
