import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

f, a = plt.subplots(10, 10, figsize=(10, 10))
for j in range(10):
    for i in range(10):
        a[j][i].imshow(np.reshape(mnist.test.images[j * 10 + i], [28, 28]))
        a[j][i].set_xticks([])
        a[j][i].set_yticks([])
plt.show()
