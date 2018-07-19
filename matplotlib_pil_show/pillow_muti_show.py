import numpy as np
from PIL import Image, ImageDraw
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

img = Image.new("RGB", (280, 280), (0, 0, 0))
draw = ImageDraw.Draw(img)
for j in range(10):
    for i in range(10):
        im = Image.fromarray(np.reshape(mnist.test.images[j * 10 + i], [28, 28]) * 255)
        im = im.convert("RGB")
        r, g, b = im.split()
        draw.bitmap((i * 28, j * 28), g, fill=(0, 255, 0))
img.show()
