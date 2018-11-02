import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


class MyMLPNet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.dp = tf.placeholder(tf.float32)

        self.W1 = tf.Variable(tf.truncated_normal([784, 512], dtype=tf.float32, stddev=tf.sqrt(1 / 512)))
        self.b1 = tf.Variable(tf.zeros([512]))

        self.W2 = tf.Variable(tf.truncated_normal([512, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.b2 = tf.Variable(tf.zeros([128]))

        self.W3 = tf.Variable(tf.truncated_normal([128, 10], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))

    def forword(self):
        self.y1 = tf.nn.sigmoid(tf.matmul(self.x, self.W1) + self.b1)
        y1_dp = tf.nn.dropout(self.y1, keep_prob=self.dp)
        self.y2 = tf.nn.sigmoid(tf.matmul(y1_dp, self.W2) + self.b2)
        y2_dp = tf.nn.dropout(self.y2, keep_prob=self.dp)
        y_out = tf.matmul(y2_dp, self.W3)
        tf.summary.histogram("y_out", y_out)
        self.y3 = tf.nn.softmax(y_out)

    def backword(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y3, labels=self.y))
        tf.summary.scalar("loss", self.loss)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        compare = tf.equal(tf.argmax(self.y3, axis=1), tf.argmax(self.y, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(compare, dtype=tf.float32))


if __name__ == '__main__':
    net = MyMLPNet()
    net.forword()
    net.backword()
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter("logs", sess.graph)
        for i in range(100000):
            xs, ys = mnist.train.next_batch(128)
            summary, loss, acc, _ = sess.run([merged, net.loss, net.accuracy, net.optimizer],
                                    feed_dict={net.x: xs, net.y: ys, net.dp: 0.8})
            writer.add_summary(summary, i)
            if i % 100 == 0:
                print("loss: {0}, accuracy: {1}".format(loss, acc))
                # test test network
                # test_xs, test_ys = mnist.train.next_batch(1)
                test_xs, test_ys = mnist.test.next_batch(1)
                test_result = sess.run([net.y3], feed_dict={net.x: test_xs, net.dp: 1.0})
                # print(type(test_result), type(test_ys))
                print("label: {0}, test: {1}".format(np.argmax(test_ys), np.argmax(test_result)))
