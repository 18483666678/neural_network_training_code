# Reference: http://blog.topspeedsnail.com/archives/10858
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random
import tensorflow as tf

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 10
            return k
        k = ord(c) - 48
        if k > 9:
            raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx == 10:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


class CNNCaptchaRecognition:

    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        self.Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout

        self.w_c1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
        self.b_c1 = tf.Variable(tf.zeros([32]))

        self.w_c2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
        self.b_c2 = tf.Variable(tf.zeros([64]))

        self.w_c3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
        self.b_c3 = tf.Variable(tf.zeros([128]))

        self.w_d = tf.Variable(tf.truncated_normal([8 * 20 * 128, 1024], stddev=0.1))
        self.b_d = tf.Variable(tf.zeros(1024))

        self.w_out = tf.Variable(tf.truncated_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN], stddev=0.1))
        self.b_out = tf.Variable(tf.zeros([MAX_CAPTCHA * CHAR_SET_LEN]))

    def forward(self):
        x = tf.reshape(self.X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

        # 3 conv layer
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, self.w_c1, strides=[1, 1, 1, 1], padding='SAME'), self.b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # conv1 = tf.nn.dropout(conv1, keep_prob)

        conv2 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(conv1, self.w_c2, strides=[1, 1, 1, 1], padding='SAME'), self.b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # conv2 = tf.nn.dropout(conv2, keep_prob)

        conv3 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(conv2, self.w_c3, strides=[1, 1, 1, 1], padding='SAME'), self.b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # conv3 = tf.nn.dropout(conv3, keep_prob)

        # Fully connected layer
        dense = tf.reshape(conv3, [-1, self.w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, self.w_d), self.b_d))
        dense = tf.nn.dropout(dense, self.keep_prob)

        self.out = tf.add(tf.matmul(dense, self.w_out), self.b_out)
        # self.out = tf.nn.softmax(self.out)

    def backward(self):
        # loss
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.Y))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.Y))
        # 最后一层用来分类的softmax和sigmoid有什么不同？
        # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        predict = tf.reshape(self.out, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__ == '__main__':
    # Initial some necessary parameters for functions
    text, image = gen_captcha_text_and_image()
    # 图像大小
    IMAGE_HEIGHT = 60
    IMAGE_WIDTH = 160
    MAX_CAPTCHA = len(text)
    print("验证码图像channel:", image.shape)  # (60, 160, 3)
    # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐
    print("验证码文本最长字符数", MAX_CAPTCHA)

    # 文本转向量
    char_set = number + ['_']  # 如果验证码长度小于4, '_'用来补齐
    CHAR_SET_LEN = len(char_set)

    # 向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每11个编码一个字符，这样顺利有，字符也有
    # vec = text2vec("5236")
    # text = vec2text(vec)
    # print(text)  # 5236
    # vec = text2vec("2468")
    # text = vec2text(vec)
    # print(text)  # 2468

    recnet = CNNCaptchaRecognition()
    recnet.forward()
    recnet.backward()
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1000000):
            batch_x, batch_y = get_next_batch(128)
            _, loss, acc = sess.run([recnet.optimizer, recnet.loss, recnet.accuracy],
                                    feed_dict={recnet.X: batch_x, recnet.Y: batch_y, recnet.keep_prob: 0.75})

            # 每100 step计算一次准确率
            if epoch % 100 == 0:
                print(epoch, loss, acc)
                # 如果准确率大于90%,保存模型,完成训练
                if acc > 0.9:
                    saver.save(sess, "./crack_capcha.ckpt", global_step=epoch)

                # Test the model
                batch_x_test, batch_y_test = get_next_batch(10)
                # saver.restore(sess, tf.train.latest_checkpoint('.'))
                out_test = sess.run(recnet.out, feed_dict={recnet.X: batch_x_test, recnet.keep_prob: 1.})

                out_test = np.argmax(np.reshape(out_test, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), axis=2)
                # print(out_test)
                for i in range(len(out_test)):
                    out_t = out_test[i].tolist()
                    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
                    j = 0
                    for n in out_t:
                        vector[j * CHAR_SET_LEN + n] = 1
                        j += 1
                    print("Label: {}, predict: {}".format(vec2text(batch_y_test[i]), vec2text(vector)))
