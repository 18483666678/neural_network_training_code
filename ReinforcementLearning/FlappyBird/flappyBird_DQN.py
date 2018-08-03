import tensorflow as tf
import numpy as np
import time
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from game import wrapped_flappy_bird

ACTIONS = 2


class Game:
    def __init__(self):
        flappyBird = wrapped_flappy_bird.GameState()
        self.experience_pool = []

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1  # one_hot: 0
        obser, reward, done = flappyBird.frame_step(do_nothing)
        obser = cv2.cvtColor(cv2.resize(obser, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, obser = cv2.threshold(obser, 1, 255, cv2.THRESH_BINARY)
        observation = np.stack((obser, obser, obser, obser), axis=2)  # shape(80, 80, 4)

        # plt.ion()
        for i in range(10000):
            if i % 5 == 0:
                index = np.random.randint(0, 2)
                action = np.zeros([2])
                action[index] = 1
            else:
                action = np.array([1, 0])

            next_obser, reward, done = flappyBird.frame_step(action)
            next_obser = self.preprocess(next_obser)
            next_observation = np.append(next_obser, observation[:, :, :3], axis=2)
            # plt.clf()
            # plt.subplot(221)
            # plt.imshow(next_observation[:, :, 0])
            # plt.subplot(222)
            # plt.imshow(next_observation[:, :, 1])
            # plt.subplot(223)
            # plt.imshow(next_observation[:, :, 2])
            # plt.subplot(224)
            # plt.imshow(next_observation[:, :, 3])
            # plt.pause(0.01)
            self.experience_pool.append([observation, reward, action, next_observation, done])
            observation = next_observation

    def get_experiences(self, batch_size):
        experiences = []
        idxs = []

        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.experience_pool))
            idxs.append(idx)
            experiences.append(self.experience_pool[idx])

        return idxs, experiences

    # preprocess raw image to 80*80 gray image
    def preprocess(slef, observation):
        observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)

        return np.reshape(observation, (80, 80, 1))

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()


class DQNet:

    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


if __name__ == '__main__':
    net = DQNet()
    net.forward()
    net.backward()

    bird = Game()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
