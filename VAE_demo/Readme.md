# Readme
Add date: 20181101

## File list
1. VAE_demo.py : This is a VAE demo from Teacher
2. VAE_cnn.py : This is a VAE demo edited by me refer to `VAE_demo.py`
3. VAE_cnn_batch_norm.py : Edited base on `VAE_cnn.py`

## Requirements
python 3  
tensorflow  
numpy  
matplotlib  

## Usage
```shell
python VAE_demo.py
python VAE_cnn.py
python VAE_cnn_batch_norm.py
```

## Recommendation
Strong recommend you use Pycharm + Anaconda to develop.


## Key points
1. `VAE_cnn.py` EncoderNet two head structure.
```python
    def forward(self, x):
        y = tf.nn.relu(tf.matmul(x, self.in_w) + self.in_b)
        mean = tf.matmul(y, self.mean_w)
        logvar = tf.matmul(y, self.logvar_w)
        return mean, logvar
```

2. `VAE_cnn.py` DncoderNet conv2d_transpose
```python
    deconv1 = tf.nn.leaky_relu(
        tf.nn.conv2d_transpose(y, self.conv1_w, output_shape=[100, 14, 14, 16], strides=[1, 2, 2, 1],
                               padding='SAME'))
    deconv2 = tf.nn.conv2d_transpose(deconv1, self.conv2_w, output_shape=[100, 28, 28, 1], strides=[1, 2, 2, 1],
                                     padding='SAME')
```

3. `VAE_cnn.py` mean and logVar
````python
    self.mean, self.logVar = self.encoder.forward(self.x)
    noise = tf.random_normal(shape=[128])
    self.std = tf.sqrt(tf.exp(self.logVar))
    y = self.std * noise + self.mean
    self.out = self.decoder.forward(y)
````

4. `VAE_cnn.py` loss function
```python
    out_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.x))
    kl_loss = tf.reduce_mean(0.5 * (-self.logVar + self.mean ** 2 + tf.exp(self.logVar) - 1))
    self.loss = out_loss + kl_loss
```