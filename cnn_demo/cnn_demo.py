import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)


class MyCNNNet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 10))

        self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1, dtype=tf.float32))
        self.conv1_b = tf.Variable(tf.zeros([16]))

        self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1, dtype=tf.float32))
        self.conv2_b = tf.Variable(tf.zeros([32]))

        self.W1 = tf.Variable(tf.truncated_normal([7 * 7 * 32, 128], stddev=tf.sqrt(1 / 128), dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros([128]))

        self.W2 = tf.Variable(tf.truncated_normal([128, 10], stddev=tf.sqrt(1 / 10), dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros([10]))

    def forword(self):
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x, self.conv1_w, [1, 1, 1, 1], "SAME") + self.conv1_b)
        self.pool1 = tf.nn.max_pool(self.conv1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.pool1, self.conv2_w, [1, 1, 1, 1], "SAME") + self.conv2_b)
        self.pool2 = tf.nn.max_pool(self.conv2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        self.flatten = tf.reshape(self.pool2, [-1, 7 * 7 * 32])
        self.y1 = tf.nn.sigmoid(tf.matmul(self.flatten, self.W1) + self.b1)
        self.y2 = tf.nn.softmax(tf.matmul(self.y1, self.W2) + self.b2)

    def backword(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y2, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        judge = tf.equal(tf.argmax(self.y2, 1), tf.argmax(self.y, 1))
        cast = tf.cast(judge, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(cast)


if __name__ == '__main__':

    net = MyCNNNet()
    net.forword()
    net.backword()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            xs, ys = mnist.train.next_batch(128)
            batch_xs = np.reshape(xs, [128, 28, 28, 1])
            acc, loss, _ = sess.run([net.accuracy, net.loss, net.optimizer], feed_dict={net.x: batch_xs, net.y: ys})

            if i % 100 == 0:
                print("loss: {0}, accuracy: {1}".format(loss, acc))
                test_xs, test_ys = mnist.train.next_batch(1)
                test_xs = np.reshape(test_xs, [1, 28, 28, 1])
                estimate_y = sess.run([net.y2], feed_dict={net.x: test_xs})
                # print(test_ys)
                # np.set_printoptions(precision=3)
                # print(np.array(estimate_y, dtype=np.float32))
                # print(type(estimate_y))
                print("labels: {0}, estimate: {1}".format(np.argmax(test_ys), np.argmax(estimate_y)))
