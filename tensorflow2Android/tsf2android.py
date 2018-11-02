# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.framework import graph_util

session = tf.Session()

matrix1 = tf.constant([[3., 3.]], name='input')
add2Mat = tf.add(matrix1, matrix1, name='output')

session.run(add2Mat)

output_graph_def = graph_util.convert_variables_to_constants(session, session.graph_def, output_node_names=['output'])

with tf.gfile.FastGFile('model/cxq.pb', mode='wb') as f:
    f.write(output_graph_def.SerializeToString())

session.close()
