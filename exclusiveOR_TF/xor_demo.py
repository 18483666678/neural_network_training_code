import numpy as np
import tensorflow as tf


class XORNet:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.lr = tf.placeholder(dtype=tf.float32)

        self.W1 = tf.Variable(tf.random_normal([2, 8], dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros([8]))

        self.W2 = tf.Variable(tf.random_normal([8, 1], dtype=tf.float32))

    def forward(self):
        self.y1 = tf.nn.sigmoid(tf.matmul(self.x, self.W1) + self.b1)
        self.y2 = tf.matmul(self.y1, self.W2)
        self.preds = tf.nn.sigmoid(self.y2)

    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y2, labels=self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)


if __name__ == '__main__':
    net = XORNet()
    net.forward()
    net.backward()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        xs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        ys = np.array([[0, 1, 1, 0]]).T
        # print(ys.shape, "\n", ys)
        for epoch in range(10000):
            if epoch < 3000:
                lr = 1
            elif epoch < 6000:
                lr = 0.1
            else:
                lr = 0.01
            loss, _ = sess.run([net.loss, net.optimizer], feed_dict={net.x: xs, net.y: ys, net.lr: lr})

            # Test the model
            if epoch % 500 == 0:
                print("loss: {0}".format(loss))
                test_xs = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
                test_ys, test_preds = sess.run([net.y2, net.preds], feed_dict={net.x: test_xs})
                print(np.array(test_ys))
                print("=========\n", np.array(test_preds))
