import gym
import time

# import mujoco_py

env = gym.make("CartPole-v0")
# env = gym.make("MountainCar-v0")
# env = gym.make("MsPacman-v0")
# env = gym.make("Pong-v0")
# env = gym.make("Pendulum-v0")
# env = gym.make("SpaceInvaders-v0")
# env = gym.make("Copy-v0")

env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.01)
