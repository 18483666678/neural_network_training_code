import tensorflow as tf
import numpy as np
import time
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from game import wrapped_flappy_bird

ACTIONS = 2
GAMMA = 0.9
LEARN_RATE = 0.01
EPISODE_MAX = 1000000
EPISODE_STEPS = 200
BATCH_SIZE = 200
INIT_EXPLORE = 0.1
MAX_MEMERY = 10000
SCOPE_EVAL = "eval"
SCOPE_TARGET = "target"


class Game:
    def __init__(self):
        self.flappyBird = wrapped_flappy_bird.GameState()
        self.experience_pool = []

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1  # one_hot: 0
        obser, reward, done = self.flappyBird.frame_step(do_nothing)
        obser = cv2.cvtColor(cv2.resize(obser, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, obser = cv2.threshold(obser, 1, 255, cv2.THRESH_BINARY)
        observation = np.stack((obser, obser, obser, obser), axis=2)  # shape(80, 80, 4)

        # plt.ion()
        for i in range(MAX_MEMERY):
            if i % 5 == 0:
                index = np.random.randint(0, 2)
                action = np.zeros([2])
                action[index] = 1
            else:
                action = np.array([1, 0])

            next_obser, reward, done = self.flappyBird.frame_step(action)
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

    def frame_step(self, action):
        obser, reward, done = self.flappyBird.frame_step(action)
        observation = self.preprocess(obser)

        return observation, reward, done

    def reset(self):
        # get the first state by doing random action and preprocess the image to 80x80x4
        index = np.random.randint(0, ACTIONS)
        action = np.zeros(ACTIONS)
        action[index] = 1
        obser, reward, done = self.flappyBird.frame_step(action)
        obser = cv2.cvtColor(cv2.resize(obser, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, obser = cv2.threshold(obser, 1, 255, cv2.THRESH_BINARY)
        observation = np.stack((obser, obser, obser, obser), axis=2)  # shape(80, 80, 4)

        return observation

    def createNewState(self, obser, next_obser):
        return np.append(next_obser, obser[:, :, :3], axis=2)


class ConvNet:

    def __init__(self, scope, actions):
        self.actions = actions
        with tf.variable_scope(scope):
            # network weights
            self.W_conv1 = self.weight_variable([8, 8, 4, 32])
            self.b_conv1 = self.bias_variable([32])

            self.W_conv2 = self.weight_variable([4, 4, 32, 64])
            self.b_conv2 = self.bias_variable([64])

            self.W_conv3 = self.weight_variable([3, 3, 64, 64])
            self.b_conv3 = self.bias_variable([64])

            self.W_fc1 = self.weight_variable([1600, 512])
            self.b_fc1 = self.bias_variable([512])

            self.W_fc2 = self.weight_variable([512, self.actions])
            self.b_fc2 = self.bias_variable([self.actions])

    # Input observation shape (N, 80, 80, 4)
    def forward(self, observation):
        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(observation, self.W_conv1, 4) + self.b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        # print("h_conv1", h_conv1)  # shape(N, 20, 20, 32)
        # print("h_pool", h_pool1)  # shape(N, 10, 10, 32)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2, 2) + self.b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)
        # print("h_conv2", h_conv2)  # shape(N, 5, 5, 64)
        # print("h_conv3", h_conv3)  # shape(N, 5, 5, 64)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        # print("h_conv3_flat", h_conv3_flat)  # shape(N, 1600)

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)

        # Q Value layer
        QValue = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

        return QValue

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def getParams(self, scope):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)


class DQNet:

    def __init__(self):
        # input layer
        self.currentState = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80, 4])
        self.observation = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80, 4])
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, 2])  # one_hot mode
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.next_observation = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80, 4])
        self.done = tf.placeholder(dtype=tf.bool, shape=[None])

        self.qNet = ConvNet(SCOPE_EVAL, ACTIONS)
        self.targetQNet = ConvNet(SCOPE_TARGET, ACTIONS)

    def forward(self, discount):
        self.eval_qs = self.qNet.forward(self.observation)
        self.eval_q = tf.expand_dims(tf.reduce_sum(tf.multiply(self.eval_qs, self.action), axis=1), axis=1)
        # print("self.eval_q", self.eval_q)  # shape(N, 1)

        self.next_qs = self.targetQNet.forward(self.next_observation)
        self.next_q = tf.expand_dims(tf.reduce_max(self.next_qs, axis=1), axis=1)

        self.target_q = tf.where(self.done, self.reward, self.reward + discount * self.next_q)

    def backward(self, learn_rate):
        self.loss = tf.reduce_mean((self.target_q - self.eval_q) ** 2)
        self.optimizer = tf.train.RMSPropOptimizer(learn_rate).minimize(self.loss)

    def getAction(self):
        action = self.qNet.forward(self.currentState)
        return tf.argmax(action, axis=1)

    def copyParams(self):
        eval_params = self.qNet.getParams(SCOPE_EVAL)
        target_params = self.targetQNet.getParams(SCOPE_TARGET)
        # print(SCOPE_EVAL, eval_params)
        # print(SCOPE_TARGET, target_params)

        # Update target network paramters
        return [tf.assign(tp, ep) for tp, ep in zip(target_params, eval_params)]


if __name__ == '__main__':
    net = DQNet()
    net.forward(GAMMA)
    net.backward(LEARN_RATE)
    copy_params = net.copyParams()
    get_action = net.getAction()

    bird = Game()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        explore = INIT_EXPLORE
        for episode in range(EPISODE_MAX):
            idxs, experiences = bird.get_experiences(BATCH_SIZE)

            observations = []
            rewards = []
            actions = []
            next_observations = []
            dones = []

            for experience in experiences:
                observations.append(experience[0])
                rewards.append([experience[1]])
                actions.append(experience[2])
                next_observations.append(experience[3])
                dones.append(experience[4])

            # # shape: (2, 80, 80, 4) (2, 1) (2, 2) (2, 80, 80, 4) (2,)
            # print(np.array(observations).shape, np.array(rewards).shape, np.array(actions).shape,
            #       np.array(next_observations).shape, np.array(dones).shape)
            # print(observations, rewards, actions, next_observations, dones)

            if episode % 10 == 0:
                print("----------------- copy param -----------------")
                sess.run(copy_params)
                # time.sleep(2)

            _loss, _ = sess.run([net.loss, net.optimizer], feed_dict={
                net.observation: observations,
                net.action: actions,
                net.reward: rewards,
                net.next_observation: next_observations,
                net.done: dones
            })

            explore -= 0.0001
            if explore < 0.0001:
                explore = 0.0001
            if episode % 100 == 0:
                print("episode: {}, loss: {}, explore: {}".format(episode, _loss, explore))

            # Test and update experiences
            run_observation = bird.reset()
            # # shape: (80, 80, 4)
            # print(run_observation.shape)
            # print(run_observation)

            for step in range(EPISODE_STEPS):

                if np.random.rand() < explore:
                    run_action = np.random.randint(0, 2)
                else:
                    run_action = sess.run(get_action, feed_dict={
                        net.currentState: [run_observation]
                    })[0]
                one_hot_action = np.zeros(ACTIONS)
                one_hot_action[run_action] = 1
                run_next_observation, run_reward, run_done = bird.frame_step(one_hot_action)
                run_next_observation = bird.createNewState(run_observation, run_next_observation)
                # print("act:\n{}, \nobser:\n{}, \nrew:\n{}, \ndo:\n{}"
                #       .format(one_hot_action, run_next_observation, run_reward, run_done))
                # time.sleep(10)
                del bird.experience_pool[0]
                bird.experience_pool.append([run_observation, run_reward, one_hot_action, run_next_observation,
                                             run_done])
                run_observation = run_next_observation
