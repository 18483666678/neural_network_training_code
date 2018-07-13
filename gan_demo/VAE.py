#VAE  p(Z) previous
#
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data')
tf.reset_default_graph()

batch_size = 64
X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
Y = tf.placeholder(dtype = tf.float32, shape=[None, 28, 28], name='X')
Y_flat = tf.reshape(Y,shape=[-1, 28*28])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 8
reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49*dec_in_channels/2

def lrelu(x,alpha=0.3):
    return tf.maxmium(x, tf.multiply(x, alpha))

def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder",reuse=None):
        X = tf.reshape(X_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5*tf.layers.dense(x, units=n_latent)
        # log(sd) 输出log的方差
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0]]))
        z = mn+tf.multiply(epsilon, tf.exp(sd))
        return z, mn, sd
        #合并
def decoder(sample_z, keep_prob):
    with tf.variable_scope("decoder",reuse=None):
        x = tf.layers.dense(sample_z,units=inputs_decoder, activation=lrelu)


