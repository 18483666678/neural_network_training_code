import tensorflow as tf
import gym
import numpy as np
import time


class QNet:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal([4, 30], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([30]))

        self.w2 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([30]))

        self.w3 = tf.Variable(tf.truncated_normal([30, 2], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([2]))

    def forward(self, observation):
        y = tf.nn.relu(tf.matmul(observation, self.w1) + self.b1)
        y = tf.nn.relu(tf.matmul(y, self.w2) + self.b2)
        y = tf.matmul(y, self.w3) + self.b3

        return y


class TargetQNet:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal([4, 30], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([30]))

        self.w2 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([30]))

        self.w3 = tf.Variable(tf.truncated_normal([30, 2], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([2]))

    def forward(self, observation):
        y = tf.nn.relu(tf.matmul(observation, self.w1) + self.b1)
        y = tf.nn.relu(tf.matmul(y, self.w2) + self.b2)
        y = tf.matmul(y, self.w3) + self.b3

        return y


class Net:
    def __init__(self):
        self.observation = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        self.action = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.next_observation = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        self.done = tf.placeholder(dtype=tf.bool, shape=[None])

        self.qNet = QNet()
        self.targetQNet = TargetQNet()

    def forward(self, discount):
        self.pre_qs = self.qNet.forward(self.observation)
        self.pre_q = tf.expand_dims(
            tf.reduce_sum(tf.multiply(tf.squeeze(tf.one_hot(self.action, 2), axis=1), self.pre_qs), axis=1), axis=1)

        self.next_qs = self.targetQNet.forward(self.next_observation)
        self.next_q = tf.expand_dims(tf.reduce_max(self.next_qs, axis=1), axis=1)

        self.target_q = tf.where(self.done, self.reward, self.reward + discount * self.next_q)

    def play(self):
        self.qs = self.qNet.forward(self.observation)
        return tf.argmax(self.qs, axis=1)

    def backward(self):
        self.loss = tf.reduce_mean((self.target_q - self.pre_q) ** 2)
        self.optimizer = tf.train.RMSPropOptimizer(0.01).minimize(self.loss)

    def copy_params(self):
        return [
            tf.assign(self.targetQNet.w1, self.qNet.w1),
            tf.assign(self.targetQNet.w2, self.qNet.w2),
            tf.assign(self.targetQNet.w3, self.qNet.w3),
            tf.assign(self.targetQNet.b1, self.qNet.b1),
            tf.assign(self.targetQNet.b2, self.qNet.b2),
            tf.assign(self.targetQNet.b3, self.qNet.b3),
        ]


class Game:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.experience_pool = []
        self.observation = self.env.reset()

        for i in range(10000):
            action = self.env.action_space.sample()
            next_observation, reward, done, info = self.env.step(action)
            self.experience_pool.append([self.observation, reward, action, next_observation, done])
            if done:
                self.observation = self.env.reset()
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
        return self.env.reset()

    def render(self):
        self.env.render()


if __name__ == '__main__':
    game = Game()

    net = Net()
    net.forward(0.9)
    net.backward()
    copy_op = net.copy_params()
    run_action_op = net.play()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        batch_size = 200
        explore = 0.1
        for k in range(10000000):
            idxs, experiences = game.get_experiences(batch_size)

            observations = []
            rewards = []
            actions = []
            next_observations = []
            dones = []

            for experience in experiences:
                observations.append(experience[0])
                rewards.append([experience[1]])
                actions.append([experience[2]])
                next_observations.append(experience[3])
                dones.append(experience[4])

            if k % 10 == 0:
                print("----------------- copy param -----------------")
                sess.run(copy_op)
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
            print("*********************", _loss, "******************", explore)

            count = 0
            run_observation = game.reset()
            for idx in idxs:
                if k > 800:
                    game.render()

                if np.random.rand() < explore:
                    run_action = np.random.randint(0, 2)
                else:
                    run_action = sess.run(run_action_op, feed_dict={
                        net.observation: [run_observation]
                    })[0]
                run_next_observation, run_reward, run_done, run_info = game.env.step(run_action)

                game.experience_pool[idx] = [run_observation, run_reward, run_action, run_next_observation,
                                             run_done]
                if run_done:
                    run_observation = game.reset()
                    count += 1
                else:
                    run_observation = run_next_observation

            print("done .........", count, k)
