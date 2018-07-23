import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


font = ImageFont.truetype(font="FreeMonoBold.ttf", size=25)


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    img = Image.new("RGB", (80, 30), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i in range(len(captcha_text)):
        draw.text((20 * i, 5), captcha_text[i], font=font, fill=(255, 255, 255))
    captcha_image = np.array(img)
    return captcha_text, captcha_image


if __name__ == '__main__':
    # 测试
    text, image = gen_captcha_text_and_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.5))
    plt.xlabel(text)
    plt.imshow(image)

    plt.show()
