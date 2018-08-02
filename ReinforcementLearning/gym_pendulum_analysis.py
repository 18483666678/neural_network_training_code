import time
import gym

# 浅析Gym中的env
# https://blog.csdn.net/u013745804/article/details/78397106
#
#         0
#         |
# pi/2 ---|--- -pi/2
#         |
#         pi
#
# e.g.: [-0.999957   -0.00927378  7.72613261]
# action = [1.9]  # 逆时针旋转
# action = [-1.9]  # 顺时针旋转
#
# self.max_speed=8
# self.max_torque=2.
# high = np.array([1., 1., self.max_speed])
# self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
# self.observation_space = spaces.Box(low=-high, high=high)
#
# cos和sin都是大于-1小于1，thdot大于-8小于8，torque大于-2小于2
#

env = gym.make("Pendulum-v0")
for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):  # 其实episode steps默认已经定义为200了，如下打印的 t
        print(t)
        env.render()
        action = env.action_space.sample()
        # action = [1.9]  # 逆时针旋转
        # action = [-1.9]  # 顺时针旋转
        action = [0.0]
        observation, reward, done, info = env.step(action)
        print(observation, reward, action)
        time.sleep(5)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
