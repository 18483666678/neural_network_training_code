import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


class MyMLPNet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        self.W1 = tf.Variable(tf.truncated_normal([784, 512], dtype=tf.float32, stddev=tf.sqrt(1 / 512)))
        self.b1 = tf.Variable(tf.zeros([512]))

        self.W2 = tf.Variable(tf.truncated_normal([512, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.b2 = tf.Variable(tf.zeros([128]))

        self.W3 = tf.Variable(tf.truncated_normal([128, 10], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))

    def forword(self):
        self.y1 = tf.nn.sigmoid(tf.matmul(self.x, self.W1) + self.b1)
        self.y2 = tf.nn.sigmoid(tf.matmul(self.y1, self.W2) + self.b2)
        self.y3 = tf.nn.softmax(tf.matmul(self.y2, self.W3))

    def backword(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y3, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        compare = tf.equal(tf.argmax(self.y3, axis=1), tf.argmax(self.y, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(compare, dtype=tf.float32))


if __name__ == '__main__':
    net = MyMLPNet()
    net.forword()
    net.backword()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            xs, ys = mnist.train.next_batch(128)
            loss, acc, _ = sess.run([net.loss, net.accuracy, net.optimizer],
                                    feed_dict={net.x: xs, net.y: ys})
            if i % 100 == 0:
                print("loss: {0}, accuracy: {1}".format(loss, acc))
                # test test network
                test_xs, test_ys = mnist.train.next_batch(1)
                test_result = sess.run([net.y3], feed_dict={net.x: test_xs})
                print("label: {0}, test: {1}".format(np.argmax(test_ys), np.argmax(test_result)))
