"""
https://blog.csdn.net/x_studying/article/details/80527498
基于DQN的五子棋算法
"""
import gym
import logging
import numpy
import random
from gym import spaces
import time
import numpy as np
import copy
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class FiveChessEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        # 棋盘大小
        self.SIZE = 4
        # 初始棋盘是0    -1表示黑棋子   1表示白棋子
        self.chessboard = [[0 for v in range(self.SIZE)] for v in range(self.SIZE)]
        self.playerJustMoved = -1
        self.viewer = None
        # self.step_count = 0
        self.n_in_row = 3

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = FiveChessEnv()
        st.playerJustMoved = self.playerJustMoved
        st.chessboard = copy.deepcopy(self.chessboard)
        return st

    def is_valid_coord(self, x, y):
        return x >= 0 and x < self.SIZE and y >= 0 and y < self.SIZE

    def is_valid_set_coord(self, x, y):
        return self.is_valid_coord(x, y) and self.chessboard[x][y] == 0

    def exchange_player(self):
        self.playerJustMoved = 0 - self.playerJustMoved
        return True

    # 返回一个有效的下棋位置
    def get_valid_pos_weights(self):
        results = []
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                if self.chessboard[x][y] == 0:
                    results.append(1)
                else:
                    results.append(0)
        return results

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        valid_pos = []
        valid = self.get_valid_pos_weights()
        for i in range(len(valid)):
            if valid[i] == 1:
                valid_pos.append(i)
        return valid_pos

    def is_n_in_row(self, state, height, width, n_in_row):
        # print(height, width, n_in_row, "\n", state)
        if np.sum(state) >= n_in_row:
            for y in range(height - n_in_row + 1):
                for x in range(width - n_in_row + 1):
                    section = state[y:y + n_in_row, x:x + n_in_row]
                    # print("========\n", section, np.sum(section))
                    if np.sum(section) >= n_in_row:
                        if np.max([np.dot(section, np.transpose(section)),
                                   np.dot(np.transpose(section), section)]) == n_in_row \
                                or np.sum(section[list(range(n_in_row)), list(range(n_in_row))]) == n_in_row \
                                or np.sum(section[list(range(n_in_row)), list(range(n_in_row - 1, -1, -1))]) == n_in_row:
                            return True
        return False

    # action 包括坐标和棋子颜色  例如：[1,3,1] 表示： 坐标（1,3），白棋
    # 输出 下一个状态，动作价值，是否结束，额外信息{}
    def step(self, action):
        '''
        #非法操作
        if not self.is_valid_set_coord(action[0],action[1]):
            return self.chessboard,-50,False,{}
        '''
        # 棋子
        self.chessboard[action[0]][action[1]] = action[2]

        # self.step_count += 1
        # # 胜负判定
        # color = action[2]

        win_reward = 1.0
        common_reward = -2.0
        draw_reward = 0

        x, y = action[0], action[1]
        x0 = 0 if x - self.n_in_row - 1 <= 0 else x - self.n_in_row - 1
        # Because of [x0, x1)
        # x1 = self.SIZE if x + self.n_in_row - 1 >= self.SIZE - 1 else x + self.n_in_row
        x1 = self.SIZE if x + self.n_in_row >= self.SIZE else x + self.n_in_row

        y0 = 0 if y - self.n_in_row - 1 <= 0 else y - self.n_in_row - 1
        y1 = self.SIZE if y + self.n_in_row >= self.SIZE else y + self.n_in_row

        section = np.array(self.chessboard)
        # print(section)
        section = section[x0:x1, y0:y1]
        # print(type(self.chessboard), self.chessboard)
        # print("==============", section.shape)
        # print(section)
        # print(section.shape[0], section.shape[1])

        action_state = np.where(section == action[2], 1, 0)
        if self.is_n_in_row(action_state, section.shape[0], section.shape[1], self.n_in_row):
            # print("Player {} ({}, {}) has {} in a row".format(action[2], action[0], action[1], self.n_in_row))
            return self.chessboard, win_reward, True, {}
        else:
            if np.sum(np.where(np.array(self.chessboard) != 0, 1, 0)) == self.SIZE * self.SIZE:
                # print("Game draw")
                return self.chessboard, draw_reward, True, {}

        return self.chessboard, common_reward, False, {}

        #
        # # 1.横向
        # count = 1
        # win = False
        #
        # i = 1
        # stop0 = False
        # stop1 = False
        #
        # while i < self.SIZE:
        #     x = action[0] + i
        #     y = action[1]
        #     # 左边
        #     if (not stop0) and self.is_valid_coord(x, y) and self.chessboard[x][y] == color:
        #         count = count + 1
        #     else:
        #         stop0 = True
        #     # 右边
        #     x = action[0] - i
        #     if (not stop1) and self.is_valid_coord(x, y) and self.chessboard[x][y] == color:
        #         count = count + 1
        #     else:
        #         stop1 = True
        #
        #     # 超过5个相同，则胜利
        #     if count >= 5:
        #         win = True
        #         break
        #
        #     # 都不相同，停止探索
        #     if stop0 and stop1:
        #         break
        #     i += 1
        #
        # if win:
        #     print('win1')
        #     return self.chessboard, win_reward, True, {}
        # # 2.纵向
        # count = 1
        # win = False
        #
        # i = 1
        # stop0 = False
        # stop1 = False
        #
        # while i < self.SIZE:
        #     x = action[0]
        #     y = action[1] + i
        #     # 左边
        #     if (not stop0) and self.is_valid_coord(x, y) and self.chessboard[x][y] == color:
        #         count = count + 1
        #     else:
        #         stop0 = True
        #     # 右边
        #     y = action[1] - i
        #     if (not stop1) and self.is_valid_coord(x, y) and self.chessboard[x][y] == color:
        #         count = count + 1
        #     else:
        #         stop1 = True
        #
        #     # 超过5个相同，则胜利
        #     if count >= 5:
        #         win = True
        #         break
        #
        #     # 都不相同，停止探索
        #     if stop0 and stop1:
        #         break
        #     i += 1
        # if win:
        #     print('win2')
        #     return self.chessboard, win_reward, True, {}
        # # 3.左斜向
        # count = 1
        # win = False
        #
        # i = 1
        # stop0 = False
        # stop1 = False
        #
        # while i < self.SIZE:
        #     x = action[0] + i
        #     y = action[1] + i
        #     # 左边
        #     if (not stop0) and self.is_valid_coord(x, y) and self.chessboard[x][y] == color:
        #         count = count + 1
        #     else:
        #         stop0 = True
        #     # 右边
        #     x = action[0] - i
        #     y = action[1] - i
        #     if (not stop1) and self.is_valid_coord(x, y) and self.chessboard[x][y] == color:
        #         count = count + 1
        #     else:
        #         stop1 = True
        #
        #     # 超过5个相同，则胜利
        #     if count >= 5:
        #         win = True
        #         break
        #
        #     # 都不相同，停止探索
        #     if stop0 and stop1:
        #         break
        #     i += 1
        # if win:
        #     print('win3')
        #     return self.chessboard, win_reward, True, {}
        #
        # # 3.右斜向
        # count = 1
        # win = False
        #
        # i = 1
        # stop0 = False
        # stop1 = False
        #
        # while i < self.SIZE:
        #     x = action[0] - i
        #     y = action[1] + i
        #     # 左边
        #     if (not stop0) and self.is_valid_coord(x, y) and self.chessboard[x][y] == color:
        #         count = count + 1
        #     else:
        #         stop0 = True
        #     # 右边
        #     x = action[0] + i
        #     y = action[1] - i
        #     if (not stop1) and self.is_valid_coord(x, y) and self.chessboard[x][y] == color:
        #         count = count + 1
        #     else:
        #         stop1 = True
        #
        #     # 超过5个相同，则胜利
        #     if count >= 5:
        #         win = True
        #         break
        #
        #     # 都不相同，停止探索
        #     if stop0 and stop1:
        #         break
        #     i += 1
        # if win:
        #     print('win4')
        #     return self.chessboard, win_reward, True, {}
        #
        # if self.step_count == self.SIZE * self.SIZE:
        #     print('draw')
        #     return self.chessboard, draw_reward, True, {}
        #
        # return self.chessboard, common_reward, False, {}

    def reset(self):
        self.chessboard = [[0 for v in range(self.SIZE)] for v in range(self.SIZE)]
        # self.step_count = 0
        return self.chessboard

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 800
        screen_height = 800
        space = 10
        width = (screen_width - space * 2) / (self.SIZE - 1)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            bg = rendering.FilledPolygon(
                [(0, 0), (screen_width, 0), (screen_width, screen_height), (0, screen_height), (0, 0)])
            bg.set_color(0.2, 0.2, 0.2)
            self.viewer.add_geom(bg)

            # 棋盘网格
            for i in range(self.SIZE):
                line = rendering.Line((space, space + i * width), (screen_width - space, space + i * width))
                line.set_color(1, 1, 1)
                self.viewer.add_geom(line)
            for i in range(self.SIZE):
                line = rendering.Line((space + i * width, space), (space + i * width, screen_height - space))
                line.set_color(1, 1, 1)
                self.viewer.add_geom(line)

            # 棋子
            self.chess = []
            for x in range(self.SIZE):
                self.chess.append([])
                for y in range(self.SIZE):
                    c = rendering.make_circle(width / 2 - 3)
                    ct = rendering.Transform(translation=(0, 0))
                    c.add_attr(ct)
                    c.set_color(0, 0, 0)
                    self.chess[x].append([c, ct])
                    self.viewer.add_geom(c)

        for x in range(self.SIZE):
            for y in range(self.SIZE):
                if self.chessboard[x][y] != 0:
                    self.chess[x][y][1].set_translation(space + x * width, space + y * width)
                    if self.chessboard[x][y] == 1:
                        self.chess[x][y][0].set_color(255, 255, 255)
                    else:
                        self.chess[x][y][0].set_color(0, 0, 0)
                else:
                    self.chess[x][y][1].set_translation(-10, -10)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # if self.state is None: return None
    # return super().render(mode)


if __name__ == '__main__':
    game = FiveChessEnv()
    game.reset()
    # Assume the first player is 1 (white)
    player = 1

    # plt.ion()
    while True:
        game.render()
        # state = game.render(mode="rgb_array")
        # plt.imshow(state)
        # plt.pause(0.01)

        # Make random action
        action = [0, 0, 0]

        # Human play
        # location = input("Player {} input: ".format(player))
        # if isinstance(location, str):  # for python3
        #     location = [int(n, 10) for n in location.split(",")]
        # action[0], action[1] = location[0], location[1]

        # Random play
        valid_pos = []
        valid = game.get_valid_pos_weights()
        for i in range(len(valid)):
            if valid[i] == 1:
                valid_pos.append(i)
        act = random.choice(valid_pos)
        action[0], action[1] = act // game.SIZE, act % game.SIZE
        action[2] = player

        # Judgement result
        _, _, win, _ = game.step(action)
        if win:
            game.render()
            # time.sleep(100)
            input("Input any to continue.")
            game.reset()

        # Exchange player
        if player == 1:
            player = -1
        else:
            player = 1

        time.sleep(0.01)
