# Readme
Add date: 20181102

## File list
binary_add_with_numpy.py : Demo from 'https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/'
binary_add_with_tf.py : Demo from 'https://blog.csdn.net/weiwei9363/article/details/78902455'  
binary_add_with_tf_M.py : Modify base on `binary_add_with_tf.py`  
LSTM_demo.py : Demo from Teacher  
LSTM_demo_M.py : Modify base on `LSTM_demo.py`, Implement MultiRNNCell  
lstm_endecode_captcha.py : Demo from Teacher, Captcha identification

## Reference
[Anyone Can Learn To Code an LSTM-RNN in Python (Part 1: RNN)](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)  
[RNN循环神经网络的直观理解：基于TensorFlow的简单RNN例子](https://blog.csdn.net/weiwei9363/article/details/78902455)  
https://github.com/jiemojiemo/TensorFlow-SimpleRNN/blob/master/TensorFlow_SimpleRNN.ipynb  

## Requirements
python 3  
tensorflow  
numpy  
matplotlib  

## Usage
```shell
python LSTM_demo.py
```

## Recommendation
Strong recommend you use Pycharm + Anaconda to develop.

## Key points
1. LSTM_demo.py
```python
    cell = tf.contrib.rnn.BasicLSTMCell(128)
    init_state = cell.zero_state(100, dtype=tf.float32)
    ys, _ = tf.nn.dynamic_rnn(cell, y, initial_state=init_state, time_major=False)
    y = ys[:, -1, :]
```

2. lstm_endecode_captcha.py
```python
    def forward(self, y):
        y = tf.expand_dims(y, axis=1)  # (128,10)==>(128,1,10)
        y = tf.tile(y, [1, 4, 1])  # (128,4,10)
```