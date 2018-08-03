import tensorflow as tf
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt


class DQNet:

    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


if __name__ == '__main__':
    net = DQNet()
    net.forward()
    net.backward()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
