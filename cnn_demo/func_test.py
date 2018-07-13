import tensorflow as tf

x = tf.constant([[[1, 3], [4, 5], [2, 7]], [[2, 4], [5, 6], [3, 8]]], dtype=tf.float32)
y1 = tf.reduce_mean(x)
y2 = tf.reduce_mean(x, 0)
y3 = tf.reduce_mean(x, 1)
y4 = tf.reduce_mean(x, 2)
# y5 = tf.reduce_mean(x, 3)



with tf.Session() as sess:
    print("=======\n", sess.run(x))
    print("=======\n", sess.run(y1))
    print("=======\n", sess.run(y2))
    print("=======\n", sess.run(y3))
    print("=======\n", sess.run(y4))
    # print("=======\n", sess.run(y5))