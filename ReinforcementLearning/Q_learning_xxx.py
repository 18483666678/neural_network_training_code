import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
# % matplotlib
# inline

# 加载实验环境
env = gym.make('FrozenLake-v0')
# Q网络解法
tf.reset_default_graph()
# 建立用于选择行为的网络的前向传播部分
inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)
# 计算预期Q值和目标Q值的差值平方和（损失值）
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)
# 训练网络
init = tf.initialize_all_variables()
# 设置超参数
y = .99
e = 0.1
num_episodes = 2000  # 为了快速设置为2000，实验调为20000时可以达到0.6的成功率
# 创建episodes中包含当前奖励值和步骤的列表
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # 初始化环境，得到第一个状态观测值
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        # Q网络
        while j < 99:
            j += 1
            # 根据Q网络和贪心算法(有随机行动的可能)选定当前的动作
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s + 1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            # 获取新的状态值和奖励值
            s1, r, d, _ = env.step(a[0])
            # 通过将新的状态值传入网络获取Q'值
            Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1 + 1]})
            # 获取最大的Q值并选定我们的动作
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y * maxQ1
            # 用目标Q值和预测Q值训练网络
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[s:s + 1], nextQ: targetQ})
            rAll += r
            s = s1
            if d == True:
                # 随着训练的进行不断减小随机行动的可能性
                e = 1. / ((i / 50) + 10)
                break
        print("++++++++++", i)
        jList.append(j)
        rList.append(rAll)
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes))
# 网络性能统计
plt.plot(rList)
plt.plot(jList)
plt.show()

# # Q-Table Learning
# import gym
# import numpy as np
#
# # 加载实验环境
# env = gym.make('FrozenLake-v0')
#
# # 集成Q表学习算法
# # 初始表（全0）
# Q = np.zeros([env.observation_space.n, env.action_space.n])
# # 设定超参数
# lr = .8
# y = .95
# num_episodes = 2000
# # 创建episodes中包含当前奖励值和步骤的列表
# rList = []
# for i in range(num_episodes):
#     # 初始化环境，得到第一个状态观测值
#     s = env.reset()
#     rAll = 0
#     d = False
#     j = 0
#     # Q表学习算法
#     while j < 99:
#         j += 1
#         # 根据Q表和贪心算法(含噪)选定当前的动作
#         a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
#         # 获取新的状态值和奖励值
#         s1, r, d, _ = env.step(a)
#         # 更新Q表
#         Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1,]) - Q[s, a])
#         rAll += r
#         s = s1
#         if d == True:
#             break
#     rList.append(rAll)
#
# print("Score over time: " + str(sum(rList) / num_episodes))
# print("Final Q-Table Values")
# print(Q)
