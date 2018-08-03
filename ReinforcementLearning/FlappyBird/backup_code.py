# flappyBird = bird.GameState()
#
# # get the first state by doing nothing and preprocess the image to 80x80x4
# do_nothing = np.zeros(ACTIONS)
# do_nothing[0] = 1
# observation, reward, done = flappyBird.frame_step(do_nothing)
# observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
# ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
# state = np.stack((observation, observation, observation, observation), axis=2)
#
# plt.ion()
# for episode in range(2000):
#     if episode % 5 == 0:
#         index = np.random.randint(0, 2)
#         action = np.zeros([2])
#         action[index] = 1
#     else:
#         action = np.array([1, 0])
#     print(index, action)
#     observation, reward, done = flappyBird.frame_step(action)
#     plt.clf()
#     plt.imshow(observation)
#     plt.pause(0.0001)
