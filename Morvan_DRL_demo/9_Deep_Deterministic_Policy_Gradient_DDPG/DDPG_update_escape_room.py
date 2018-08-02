"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'

game_rewards = np.array(
    [[-100, -100, -100, -100, 0, -100],
     [-100, -100, -100, 0, -100, 100],
     [-100, -100, -100, 0, -100, -100],
     [-100, 0, 0, -100, 0, -100],
     [0, -100, -100, 0, -100, 100],
     [-100, 0, -100, -100, 0, 100]]
)


class Game:

    def __init__(self):
        self.experience_pool = []
        self.observation = np.random.choice(6)

        for i in range(10000):
            action = np.random.choice(6)
            next_observation, reward, done = action, game_rewards[self.observation, action], action == 5
            self.experience_pool.append([self.observation, reward / 100, action, next_observation, done])
            if done:
                self.observation = np.random.choice(6)
            else:
                self.observation = next_observation

    def get_experiences(self, batch_size):
        experiences = []
        idxs = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.experience_pool))
            idxs.append(idx)
            experiences.append(self.experience_pool[idx])

        return idxs, experiences

    def reset(self):
        return np.random.choice(6)

    def render(self):
        pass


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim):
        # shape(self.memory) : [10000, 1 * 2 + 1 + 1]
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.int32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.int32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.int32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            self.action = tf.argmax(self.a, axis=1, name='scaled_a')
            a_ = self._build_a(self.S_, scope='target', trainable=False)
            # print(self.a, a_)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)
            # print(q, q_)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        # print(self.ae_params)
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=self.ce_params)

        self.a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params + self.ce_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        a = self.sess.run(self.action, {self.S: [s]})
        # print(a)
        # a = a[0]
        return a

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # print("learn:bs {}, ba {}, br {}, bs_ {}".format(bs, ba[:, 0], br, bs_))
        ba_one_hot = np.zeros((BATCH_SIZE, 6))
        # print(ba_one_hot)
        # print(np.random.choice(6, BATCH_SIZE))
        # print(ba[:, 0])
        ba_one_hot[np.arange(BATCH_SIZE), ba[:, 0]] = 1

        a_loss, _ = self.sess.run([self.a_loss, self.atrain], {self.S: bs})
        c_loss, _ = self.sess.run([self.td_error, self.ctrain],
                                  {self.S: bs, self.a: ba_one_hot, self.R: br, self.S_: bs_})

        return a_loss, c_loss

    def store_transition(self, s, a, r, s_):
        transition = np.hstack(([s], [a], [r], [s_]))
        # print("zeng>> trans:", s, a, r, s_, transition)
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            observ = tf.squeeze(tf.one_hot(s, 6), axis=1)

            net = tf.layers.dense(observ, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim * 6, activation=tf.nn.softmax, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            observ = tf.squeeze(tf.one_hot(s, 6), axis=1)
            # action = tf.expand_dims(a, axis=1)
            # action = tf.squeeze(tf.one_hot(a, 6), axis=1)

            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim * 6, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim * 6, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(observ, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


###############################  training  ####################################

game = Game()
a_dim = 1
s_dim = 1

ddpg = DDPG(a_dim, s_dim)

explore = 0.1  # control exploration
t1 = time.time()

for i in range(MAX_EPISODES):
    s = [game.reset()]
    ep_reward = 0

    explore -= 0.0001  # decay the action randomness
    if explore < 0.0001:
        explore = 0.0001

    for j in range(MAX_EP_STEPS):

        # Add exploration noise
        if np.random.rand() < explore:
            a = [np.random.choice(6)]  # add randomness to action selection for exploration
            # print("zeng>> rand a: ", a, type(a))

        else:
            a = ddpg.choose_action(s)
            # print("zeng>> choose a: ", a, type(a))
        s_, r, done = a, game_rewards[s, a], a == 5

        # print("zeng>> store, ", s, a, r / 100, s_)
        ddpg.store_transition(s, a, r / 100, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            a_loss, c_loss = ddpg.learn()
            if i % 10 == 0 and j == MAX_EP_STEPS - 1:
                print("a_loss: {}, c_loss: {}".format(a_loss, c_loss))

            if i >= MAX_EPISODES - 50:
                time.sleep(1)
                print("{}{}==>".format(s[0], a), end="")
                if s == [5]:
                    print("done\n")

        s = s_
        ep_reward += r / 100
        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % explore, )
            # if ep_reward > -300:
            #     RENDER = True
            # break
print('Running time: ', time.time() - t1)
