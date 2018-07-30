import tensorflow as tf
import gym
import numpy as np
import time


class QNet:

    def __init__(self):
        with tf.variable_scope("ActorNet"):
            self.a_w1 = tf.Variable(tf.truncated_normal([6, 30], stddev=0.1))
            self.a_b1 = tf.Variable(tf.zeros([30]))

            self.a_w2 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
            self.a_b2 = tf.Variable(tf.zeros([30]))

            self.a_w3 = tf.Variable(tf.truncated_normal([30, 6], stddev=0.1))
            self.a_b3 = tf.Variable(tf.zeros([6]))

        with tf.variable_scope("CriticNet"):
            self.c_w1 = tf.Variable(tf.truncated_normal([6, 30], stddev=0.1))
            self.c_b1 = tf.Variable(tf.zeros([30]))

            self.c_w2 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
            self.c_b2 = tf.Variable(tf.zeros([30]))

            self.c_w3 = tf.Variable(tf.truncated_normal([30, 6], stddev=0.1))
            self.c_b3 = tf.Variable(tf.zeros([6]))

    def actor_forward(self, observation):
        y = tf.nn.relu(tf.matmul(observation, self.a_w1) + self.a_b1)
        y = tf.nn.relu(tf.matmul(y, self.a_w2) + self.a_b2)
        y = tf.matmul(y, self.a_w3) + self.a_b3

        # y = tf.round(y)
        # y = tf.abs(y)
        # y = tf.cast(y % 6, tf.int32)
        y = tf.argmax(y, axis=1)

        return y

    def critic_forward(self, observation):
        y = tf.nn.relu(tf.matmul(observation, self.c_w1) + self.c_b1)
        y = tf.nn.relu(tf.matmul(y, self.c_w2) + self.c_b2)
        y = tf.matmul(y, self.c_w3) + self.c_b3

        return y

    def getParams(self, name):
        if name == "ActorNet":
            return tf.get_collection(tf.GraphKeys.VARIABLES, scope="ActorNet") + tf.get_collection(
                tf.GraphKeys.VARIABLES, scope="CriticNet")
        elif name == "CriticNet":
            return tf.get_collection(tf.GraphKeys.VARIABLES, scope="CriticNet")


class TargetQNet:

    def __init__(self):
        with tf.variable_scope("ActorNet"):
            self.a_w1 = tf.Variable(tf.truncated_normal([6, 30], stddev=0.1))
            self.a_b1 = tf.Variable(tf.zeros([30]))

            self.a_w2 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
            self.a_b2 = tf.Variable(tf.zeros([30]))

            self.a_w3 = tf.Variable(tf.truncated_normal([30, 6], stddev=0.1))
            self.a_b3 = tf.Variable(tf.zeros([6]))

        with tf.variable_scope("CriticNet"):
            self.c_w1 = tf.Variable(tf.truncated_normal([6, 30], stddev=0.1))
            self.c_b1 = tf.Variable(tf.zeros([30]))

            self.c_w2 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
            self.c_b2 = tf.Variable(tf.zeros([30]))

            self.c_w3 = tf.Variable(tf.truncated_normal([30, 6], stddev=0.1))
            self.c_b3 = tf.Variable(tf.zeros([6]))

    def actor_forward(self, observation):
        y = tf.nn.relu(tf.matmul(observation, self.a_w1) + self.a_b1)
        y = tf.nn.relu(tf.matmul(y, self.a_w2) + self.a_b2)
        y = tf.matmul(y, self.a_w3) + self.a_b3

        # y = tf.round(y)
        # y = tf.abs(y)
        # y = tf.cast(y % 6, tf.int32)
        y = tf.argmax(y, axis=1)

        return y

    def critic_forward(self, observation):
        y = tf.nn.relu(tf.matmul(observation, self.c_w1) + self.c_b1)
        y = tf.nn.relu(tf.matmul(y, self.c_w2) + self.c_b2)
        y = tf.matmul(y, self.c_w3) + self.c_b3

        return y

    def getParams(self, name):
        if name == "ActorNet":
            return tf.get_collection(tf.GraphKeys.VARIABLES, scope="ActorNet") + tf.get_collection(
                tf.GraphKeys.VARIABLES, scope="CriticNet")
        elif name == "CriticNet":
            return tf.get_collection(tf.GraphKeys.VARIABLES, scope="CriticNet")


class Net:

    def __init__(self):
        self.observation = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.action = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.next_observation = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.done = tf.placeholder(dtype=tf.bool, shape=[None])

        self.qNet = QNet()
        self.targetQNet = TargetQNet()

    def forward(self, discount):
        observ = tf.squeeze(tf.one_hot(self.observation, 6), axis=1)
        self.pre_qs = self.qNet.critic_forward(observ)
        self.pre_q = tf.expand_dims(
            tf.reduce_sum(tf.multiply(tf.squeeze(tf.one_hot(self.action, 6), axis=1), self.pre_qs), axis=1), axis=1)

        self.pre_action = self.qNet.actor_forward(observ)
        self.pre_action = tf.expand_dims(self.pre_action, axis=1)
        self.pre_q_a = tf.expand_dims(
            tf.reduce_sum(tf.multiply(tf.squeeze(tf.one_hot(self.pre_action, 6), axis=1), self.pre_qs), axis=1), axis=1)

        next_observ = tf.squeeze(tf.one_hot(self.next_observation, 6), axis=1)
        self.next_action = self.targetQNet.actor_forward(next_observ)
        self.next_action = tf.expand_dims(self.next_action, axis=1)
        self.next_qs = self.targetQNet.critic_forward(next_observ)
        self.next_q_a = tf.expand_dims(
            tf.reduce_sum(tf.multiply(tf.squeeze(tf.one_hot(self.next_action, 6), axis=1), self.next_qs), axis=1),
            axis=1)

        self.target_q = tf.where(self.done, self.reward, self.reward + discount * self.next_q_a)

    def play(self):
        observ = tf.squeeze(tf.one_hot(self.observation, 6), axis=1)
        self.qs = self.qNet.critic_forward(observ)
        return tf.argmax(self.qs, axis=1)

    def backward(self):
        self.critic_loss = tf.reduce_mean((self.target_q - self.pre_q) ** 2)
        self.critic_opt = tf.train.RMSPropOptimizer(0.01).minimize(self.critic_loss,
                                                                   var_list=self.qNet.getParams("CriticNet"))

        self.actor_loss = tf.reduce_mean(tf.multiply(tf.cast(self.pre_action, tf.float32), self.pre_q_a))
        self.actor_opt = tf.train.RMSPropOptimizer(0.01).minimize(self.actor_loss,
                                                                  var_list=self.qNet.getParams("ActorNet"))

    def copy_params(self):
        return [
            tf.assign(self.targetQNet.a_w1, self.qNet.a_w1),
            tf.assign(self.targetQNet.a_w2, self.qNet.a_w2),
            tf.assign(self.targetQNet.a_w3, self.qNet.a_w3),
            tf.assign(self.targetQNet.a_b1, self.qNet.a_b1),
            tf.assign(self.targetQNet.a_b2, self.qNet.a_b2),
            tf.assign(self.targetQNet.a_b3, self.qNet.a_b3),

            tf.assign(self.targetQNet.c_w1, self.qNet.c_w1),
            tf.assign(self.targetQNet.c_w2, self.qNet.c_w2),
            tf.assign(self.targetQNet.c_w3, self.qNet.c_w3),
            tf.assign(self.targetQNet.c_b1, self.qNet.c_b1),
            tf.assign(self.targetQNet.c_b2, self.qNet.c_b2),
            tf.assign(self.targetQNet.c_b3, self.qNet.c_b3),
        ]


game_rewards = np.array(
    [[-100, -100, -100, -100, 0, -100],
     [-100, -100, -100, 0, -100, 100],
     [-100, -100, -100, 0, -100, -100],
     [-100, 0, 0, -100, 0, -100],
     [0, -100, -100, 0, -100, 100],
     [-100, 0, -100, -100, 0, 100]]
)
# game_rewards = np.array(
#     [[-1, -1, -1, -1, 0, -1],
#      [-1, -1, -1, 0, -1, 100],
#      [-1, -1, -1, 0, -1, -1],
#      [-1, 0, 0, -1, 0, -1],
#      [0, -1, -1, 0, -1, 100],
#      [-1, 0, -1, -1, 0, 100]]
# )
gamma = 0.8


class Game:

    def __init__(self):
        self.experience_pool = []
        self.observation = np.random.choice(6)

        for i in range(1000):
            action = np.random.choice(6)
            next_observation, reward, done = action, game_rewards[self.observation, action], action == 5
            self.experience_pool.append([self.observation, reward, action, next_observation, done])
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


if __name__ == '__main__':
    game = Game()

    net = Net()
    net.forward(gamma)
    net.backward()
    copy_op = net.copy_params()
    run_action_op = net.play()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        batch_size = 200

        explore = 0.1
        for k in range(1000000):
            idxs, experiences = game.get_experiences(batch_size)

            # print(idxs)
            # print(experiences)

            observations = []
            rewards = []
            actions = []
            next_observations = []
            dones = []

            for experience in experiences:
                observations.append([experience[0]])
                rewards.append([experience[1]])
                actions.append([experience[2]])
                next_observations.append([experience[3]])
                dones.append(experience[4])

            # print(observations, "\n", rewards, "\n", actions, "\n", next_observations, "\n", dones)
            if k % 10 == 0:
                # print("--------- copy param ---------")
                sess.run(copy_op)
                # time.sleep(2)

            c_loss, _ = sess.run([net.critic_loss, net.critic_opt], feed_dict={
                net.observation: observations,
                net.action: actions,
                net.reward: rewards,
                net.next_observation: next_observations,
                net.done: dones
            })

            pre_action, pre_q_a, a_loss, _ = sess.run([net.pre_action, net.pre_q_a, net.actor_loss, net.actor_opt],
                                                      feed_dict={
                                                          net.observation: observations,
                                                      })

            explore -= 0.0001
            if explore < 0.0001:
                explore = 0.0001

            if k % 100 == 0:
                # print("\npre_action:\n", pre_action)
                # print("\npre_q_a:\n", pre_q_a)
                print("episode:{}, c_loss: {}, a_loss: {}, explore: {}".format(k, c_loss, a_loss, explore))

            count = 0
            run_observation = game.reset()
            for idx in idxs:
                if k % 100 == 0:
                    print("{} ==>".format(run_observation), end="")

                if np.random.rand() < explore:
                    run_action = np.random.randint(0, 6)
                else:
                    run_action = sess.run(run_action_op, feed_dict={
                        net.observation: [[run_observation]]
                    })[0]

                run_next_observation, run_reward, run_done = run_action, game_rewards[
                    run_observation, run_action], run_action == 5

                game.experience_pool[idx] = [run_observation, run_reward, run_action, run_next_observation, run_done]
                if run_done:
                    run_observation = game.reset()
                    if k % 100 == 0:
                        print("5")
                else:
                    run_observation = run_next_observation
