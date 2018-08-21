import numpy as np

width = 6
height = 10
n_in_row = 5
player1 = 1
player2 = -1


def is_n_in_row(state, height, width, n_in_row):
    if np.sum(state) >= n_in_row:
        for h in range(height - n_in_row + 1):
            for w in range(width - n_in_row + 1):
                section = state[h:h + n_in_row, w:w + n_in_row]
                # print("========\n", section, np.sum(section))
                if np.sum(section) >= n_in_row:
                    if np.max([np.dot(section, np.transpose(section)),
                               np.dot(np.transpose(section), section)]) == n_in_row \
                            or np.sum(section[list(range(n_in_row)), list(range(n_in_row))]) == n_in_row \
                            or np.sum(section[list(range(n_in_row)), list(range(n_in_row - 1, -1, -1))]) == n_in_row:
                        print("Range:\n  H: [{} ~ {}), W: [{} ~ {})".format(h, h + n_in_row, w, w + n_in_row))
                        return True
    return False


def getResult(state):
    state_1 = np.where(state == player1, 1, 0)
    state_2 = np.where(state == player2, 1, 0)
    if is_n_in_row(state_1, height, width, n_in_row):
        print("Player {} has {} in a row".format(player1, n_in_row))
    elif is_n_in_row(state_2, height, width, n_in_row):
        print("Player {} has {} in a row".format(player2, n_in_row))
    else:
        print(np.sum([state_1, state_2]), height * width)


# board = np.zeros((width, height))
# board = np.ones((width, height))
board = np.random.randint(-1, 2, height * width).reshape(height, width)
print(board)
getResult(board)

# chess = board.tolist()
# print(np.where(np.array(chess) != 0, 1, 0))
# print(np.sum(np.where(np.array(chess) != 0, 1, 0)))

#
# Below is some test code for debug
#
# print(list(range(5, -1, -1)))
# for i in range(5, -1, -1):
#     print(i)

# one = np.ones(5)
# print(one)
# print(one.reshape(-1, 1))
# list4 = [[1, 1, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
# list4 = [[1, 1, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
# print(np.transpose(list4)[])
# board = np.random.randint(-1, 1, 5 * 5).reshape(5, 5)
# board = np.array(list4)[::-1]
# print(board)
# print("=======")
# xx = np.dot(board, np.transpose(board))
# yy = np.dot(np.transpose(board), board)
# print(xx, np.max(xx))
# print("=======")
# print(yy, np.max(yy))

# list1 = [[0, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]]
# list2 = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
# list3 = [[0, 0, 1, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
# list4 = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]

# board = np.array(list1)
# print(board)
# print(np.dot(np.transpose(list1), list1))
# print(np.dot(one, list2))
# print(np.dot(one, list3))
# print(np.dot(one, list4))

# print(np.dot(one, list1) >= one)
# if np.dot(np.dot(one, list1),one.reshape(-1, 1)):
#     print("zeng>>")
# for i in list1, list2, list3, list4:
#     print()
#     # print(i)
#     # if np.dot(one, i) >= one:
#     #     print(i)
#
# print(section[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
# print(np.sum(section[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]))
# print(section[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
# print(np.sum(section[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]))
# print("hello===============")


# Warning: This is a issue method, it can not think overall
#
# def is_n_in_row(board):
#     index = np.where(board.flatten() > 0)
#     # print(index[0], type(index[0]), len(index[0]))
#     valid_index = [x for x in index[0] if x % width <= width - n_in_row and x // height <= height - n_in_row]
#     # print(valid_index)
#     if len(valid_index) >= n_in_row:
#         for x in valid_index:
#             section = board[x % width:x % width + n_in_row, x // height: x // height + n_in_row]
#             # print("========\n", section, np.sum(section))
#             if np.sum(section) >= n_in_row:
#                 if np.max([np.dot(section, np.transpose(section)),
#                            np.dot(np.transpose(section), section)]) == n_in_row \
#                         or np.sum(section[list(range(n_in_row)), list(range(n_in_row))]) == n_in_row \
#                         or np.sum(section[list(range(n_in_row)), list(range(n_in_row - 1, -1, -1))]) == n_in_row:
#                     return True
#     return False
