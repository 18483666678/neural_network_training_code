import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


# print(np.min(mnist.train.images[0]))

class EncoderNet:
    def __init__(self):
        self.in_w = tf.Variable(tf.truncated_normal(shape=[784, 100], stddev=0.1))
        self.in_b = tf.Variable(tf.zeros([100]))

        self.logvar_w = tf.Variable(tf.truncated_normal(shape=[100, 128], stddev=0.1))
        self.mean_w = tf.Variable(tf.truncated_normal(shape=[100, 128], stddev=0.1))

    def forward(self, x):
        y = tf.nn.relu(tf.matmul(x, self.in_w) + self.in_b)
        mean = tf.matmul(y, self.mean_w)
        logvar = tf.matmul(y, self.logvar_w)
        return mean, logvar


class DecoderNet:
    def __init__(self):
        self.in_w = tf.Variable(tf.truncated_normal(shape=[128, 100], stddev=0.1))
        self.in_b = tf.Variable(tf.zeros([100]))

        self.out_w = tf.Variable(tf.truncated_normal(shape=[100, 784], stddev=0.1))

    def forward(self, x):
        y = tf.nn.relu(tf.matmul(x, self.in_w) + self.in_b)
        return tf.matmul(y, self.out_w)


class VaeNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

        self.encoderNet = EncoderNet()
        self.decoderNet = DecoderNet()

        self.forward()
        self.backward()

    def forward(self):
        self.mean, self.logvar = self.encoderNet.forward(self.x)
        self.noise = tf.random_normal(shape=[128])
        self.std = tf.sqrt(tf.exp(self.logvar))
        self.y = self.std * self.noise + self.mean
        self.output = self.decoderNet.forward(self.y)

    def decode(self):
        noise = tf.random_normal(shape=[1, 128])
        return self.decoderNet.forward(noise)

    def backward(self):
        self.output_loss = tf.reduce_mean((self.output - self.x) ** 2)
        self.kl_loss = tf.reduce_mean(0.5 * (-self.logvar + self.mean ** 2 + tf.exp(self.logvar) - 1))
        self.loss = self.output_loss + self.kl_loss

        self.opt = tf.train.AdamOptimizer(0.0001).minimize(self.loss)


if __name__ == '__main__':
    net = VaeNet()
    decode_out = net.decode()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        plt.ion()
        for epoch in range(10000000):
            xs, _ = mnist.train.next_batch(100)

            _loss, _ = sess.run([net.loss, net.opt], feed_dict={net.x: xs})

            if epoch % 1000 == 0:
                print(_loss)
                test_output = sess.run(decode_out)
                img = np.reshape(test_output[0], (28, 28))
                # img = np.array(np.clip(img*255,0,255), dtype=np.int8)
                # print(img)
                plt.imshow(img)
                plt.pause(1)
