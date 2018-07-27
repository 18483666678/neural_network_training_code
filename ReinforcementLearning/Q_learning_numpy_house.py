# Reference: Q-学习：强化学习
# https://blog.csdn.net/u013405574/article/details/50903987
#
# Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
import numpy as np
import pandas as pd

# rewards = pd.DataFrame(
#     [[-1, -1, -1, -1, 0, -1],
#      [-1, -1, -1, 0, -1, 100],
#      [-1, -1, -1, 0, -1, -1],
#      [-1, 0, 0, -1, 0, -1],
#      [0, -1, -1, 0, -1, 100],
#      [-1, 0, -1, -1, 0, 100]], index=['a', 'b', 'c', 'd', 'e', 'f'], columns=[0, 1, 2, 3, 4, 5]
# )
# print(rewards)
# print(rewards[5]['b'])  # [columns][index]

rewards = np.array(
    [[-1, -1, -1, -1, 0, -1],
     [-1, -1, -1, 0, -1, 100],
     [-1, -1, -1, 0, -1, -1],
     [-1, 0, 0, -1, 0, -1],
     [0, -1, -1, 0, -1, 100],
     [-1, 0, -1, -1, 0, 100]]
)
vaild_actions = [[4], [3, 5], [3], [1, 2, 4], [0, 3, 5], [1, 4, 5]]
q_table = np.zeros([6, 6])

gamma = 0.8

# print(reward)
# print(q_table)
# print(vaild_action)
# print(np.random.choice(6)

for epoch in range(10000):
    state = np.random.choice(6)
    while True:
        action = np.random.choice(vaild_actions[state])
        reward = rewards[state, action]
        # print("state: {}, action: {}, reward: {}".format(state, action, reward))

        max_nxq_val = 0
        for i in vaild_actions[action]:
            nxq_val = q_table[action][i]
            # print(i, nxq_val)
            if max_nxq_val < nxq_val:
                max_nxq_val = nxq_val
        # print("The max of next q value:", max_nxq_val)

        q_val = reward + gamma * max_nxq_val
        # print("The current q value:", q_val)
        q_table[state, action] = q_val

        if epoch % 1000 == 0:
            print("epoch: {}, state: {}, action: {}, reward: {}".format(epoch, state, action, reward))
            print(pd.DataFrame(np.round(q_table / 5), index=[0, 1, 2, 3, 4, 5], columns=[0, 1, 2, 3, 4, 5],
                               dtype=np.int32))

        state = action
        if state == 5:
            break
