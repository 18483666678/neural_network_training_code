import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


class LstmNet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        # self.in_w = tf.Variable(tf.truncated_normal(shape=[28, 128], stddev=0.1))
        # self.in_b = tf.Variable(tf.zeros([128]))

        self.out_w = tf.Variable(tf.truncated_normal(shape=[256, 10], stddev=0.1))
        self.out_b = tf.Variable(tf.zeros([10]))

        self.forward()
        self.backward()

    def forward(self):
        y = tf.reshape(self.x, shape=[-1, 28, 28])
        # y = tf.nn.relu(tf.matmul(y, self.in_w) + self.in_b)
        #
        # y = tf.reshape(y, shape=[-1, 28, 128])

        # cell = tf.contrib.rnn.BasicLSTMCell(128)
        cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(layer) for layer in (128, 256)])
        init_state = cell.zero_state(100, dtype=tf.float32)
        ys, _ = tf.nn.dynamic_rnn(cell, y, initial_state=init_state, time_major=False)
        y = ys[:, -1, :]

        self.output = tf.nn.softmax(tf.matmul(y, self.out_w) + self.out_b)

    def backward(self):
        self.loss = tf.reduce_mean((self.output - self.y) ** 2)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def test(self):
        pass


if __name__ == '__main__':
    net = LstmNet()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(10000000):
            xs, ys = mnist.train.next_batch(100)

            _loss, _ = sess.run([net.loss, net.opt], feed_dict={net.x: xs, net.y: ys})

            if epoch % 100 == 0:
                print(_loss)
                test_xs, test_ys = mnist.test.next_batch(100)
                test_output = sess.run(net.output, feed_dict={net.x: test_xs})

                test_y = np.argmax(test_ys, axis=1)
                test_out = np.argmax(test_output, axis=1)
                print(np.mean(np.array(test_y == test_out, dtype=np.float32)))
