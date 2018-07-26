import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


class EncoderNet:

    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal(shape=[784, 512], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([512]))

        self.logvar_w = tf.Variable(tf.truncated_normal(shape=[512, 128], stddev=0.1))
        self.mean_w = tf.Variable(tf.truncated_normal(shape=[512, 128], stddev=0.1))

    def forward(self, x):
        y = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)

        logVar = tf.matmul(y, self.logvar_w)
        mean = tf.matmul(y, self.mean_w)

        return mean, logVar


class DecoderNet:

    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal(shape=[128, 512], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([512]))

        self.w2 = tf.Variable(tf.truncated_normal(shape=[512, 784], stddev=0.1))

    def forward(self, x):
        y = tf.nn.leaky_relu(tf.matmul(x, self.w1) + self.b1)
        y = tf.matmul(y, self.w2)

        return y


class VAENet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

        self.encoder = EncoderNet()
        self.decoder = DecoderNet()

    def forward(self):
        self.mean, self.logVar = self.encoder.forward(self.x)

        noise = tf.random_normal(shape=[128])
        self.std = tf.sqrt(tf.exp(self.logVar))
        y = self.std * noise + self.mean
        self.out = self.decoder.forward(y)

    def backward(self):
        out_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.x))
        kl_loss = tf.reduce_mean(0.5 * (-self.logVar + self.mean ** 2 + tf.exp(self.logVar) - 1))
        self.loss = out_loss + kl_loss

        self.opt = tf.train.AdamOptimizer(0.0002).minimize(self.loss)


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
