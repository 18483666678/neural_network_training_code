import tensorflow as tf
import os
import matplotlib.image as imgplt
import numpy as np

batch_size = 10
weight = 120
high = 60
channel = 3


class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[batch_size, high, weight, channel])
        self.y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 4, 10])
        self.encode = Encoder()
        self.decode = Decoder()

    def forward(self):
        y = self.encode.forward(self.x)
        self.output = self.decode.forward(y)

    def backward(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)


class Encoder:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal(shape=(60 * 3, 128)))
        self.b1 = tf.Variable([128], dtype=tf.float32)

    def forward(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [batch_size * 120, 60 * 3])
        x = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        x = tf.reshape(x, [batch_size, 120, 128])
        with tf.variable_scope('encode'):
            cell = tf.contrib.rnn.BasicLSTMCell(128)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            encode_outputs, encode_final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, time_major=False)
            x = tf.transpose(encode_outputs, [1, 0, 2])[-1]
            return x


class Decoder:
    def __init__(self):
        self.w2 = tf.Variable(tf.truncated_normal(shape=(128, 10)))
        self.b2 = tf.Variable([10], dtype=tf.float32)

    def forward(self, y):
        y = tf.expand_dims(y, axis=1)  # (128,10)==>(128,1,10)
        y = tf.tile(y, [1, 4, 1])  # (128,4,10)
        with tf.variable_scope('decode'):
            cell = tf.contrib.rnn.BasicLSTMCell(128)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            decode_outputs, decode_final_state = tf.nn.dynamic_rnn(cell, y, initial_state=init_state, time_major=False)
            y = tf.reshape(decode_outputs, [batch_size * 4, 128])
            y = tf.nn.softmax(tf.matmul(y, self.w2) + self.b2)  # y(40,128) w2(128,10)
            y = tf.reshape(y, [batch_size, 4, 10])
            return y


class sample:
    def __init__(self):
        self.dataset = []
        for filename in os.listdir("D:\code"):
            x = imgplt.imread(os.path.join("D:\code", filename)) / 255. - 0.5
            y = filename.split(".")[0]
            y = self.__one_hot(y)
            self.dataset.append([x, y])

    def get_batch(self, size):
        xs = []
        ys = []
        for _ in range(size):
            index = np.random.randint(0, len(self.dataset))
            xs.append((self.dataset[index][0]))
            ys.append((self.dataset[index][1]))
        return xs, ys

    def __one_hot(self, x):
        z = np.zeros(shape=(4, 10))
        for i in range(4):
            index = int(x[i])
            z[i][index] += 1
        return z


if __name__ == '__main__':
    sample = sample()
    net = Net()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        xs, ys = sample.get_batch(batch_size)
        sess.run(net.opt, feed_dict={net.x: xs, net.y: ys})
