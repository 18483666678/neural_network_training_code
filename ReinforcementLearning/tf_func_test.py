# self.pre_q = tf.expand_dims(
#     tf.reduce_sum(tf.multiply(tf.squeeze(tf.one_hot(self.action, 2), axis=1), self.pre_qs), axis=1), axis=1)
#
# self.next_q = tf.expand_dims(tf.reduce_max(self.next_qs, axis=1), axis=1)
#
import tensorflow as tf

action = [[1], [0], [1]]
q_val = [[3, 4], [5, 6], [7, 8]]

# action = tf.constant(actions)
# q_val = tf.constant(q_vals)

one_action = tf.one_hot(action, 2)
print(one_action)
one_action = tf.expand_dims(one_action, 2)
print(one_action)

squ_action = tf.squeeze(one_action)
print(squ_action)

sel_val = tf.multiply(squ_action, q_val)
sum_val = tf.reduce_sum(sel_val, axis=1)
exp_val = tf.expand_dims(sum_val, axis=1)

###### cast ######
x = tf.constant([[0.9], [2.5], [2.3], [1.5], [-4.5]])
y = tf.round(x)  # [ [1.0], [2.0], [2.0], [2.0], [-4.0] ]
y = tf.abs(y)  # [ [1.0], [2.0], [2.0], [2.0], [4.0] ]
y = tf.cast(y, tf.int32)  # [ [1], [2], [2], [2], [4] ]
y_div = tf.cast(y % 3, tf.int32)  # [ [1], [2], [2], [2], [1] ]

print(y)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(one_action))
    print(sess.run(squ_action))
    print(sess.run(sel_val))
    print(sess.run(sum_val))
    print(sess.run(exp_val))
    print("###### cast ######")
    print(sess.run(y))
    print(sess.run(y_div))
