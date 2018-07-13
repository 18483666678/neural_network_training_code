import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1, 1, 0]]).T
syn0 = 2 * np.random.random((2, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1
plt.figure(1)
plt.ion()

for j in range(60000):
    l1 = 1 / (1 + np.exp(-(np.dot(X, syn0))))
    l2 = 1 / (1 + np.exp(-(np.dot(l1, syn1))))
    l2_delta = (y - l2) * (l2 * (1 - l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1 - l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)

    if j % 100 == 0:
        X_test = np.array([[0, 1], [0, 0], [1, 1], [1, 0]])
        l1_test = 1 / (1 + np.exp(-(np.dot(X_test, syn0))))
        l2_test = 1 / (1 + np.exp(-(np.dot(l1_test, syn1))))
        print(l2.flatten())
        print(l2_test.flatten())

        plt.clf()
        plt.subplot(221)
        plt.title("l1 out histogram")
        plt.hist(l1.flatten())

        plt.subplot(222)
        plt.title("l2 out histogram")
        plt.hist((y - l2).flatten())

        plt.subplot(223)
        plt.title("l2 out")
        plt.plot(l2.flatten(), ".")

        plt.subplot(224)
        plt.title("l2 loss out")
        plt.plot((y - l2).flatten(), ".")
        plt.pause(0.5)

plt.ioff()

# status_score = np.array([[[80, 88], [82, 81], [84, 75], [86, 83], [75, 81]]])
# status_score2 = np.array([[[81, 88], [83, 81], [85, 75], [87, 83], [76, 81]]])
#
# print(np.stack((status_score, status_score2)))

# arr1 = np.array(np.arange(3, 8), ndmin=2)
# arr = np.array(np.arange(3, 8), ndmin=2)
# print(arr1 * arr)
# print("++++++++++++++++")
# print(arr)

# arr = np.random.random((3,4))
# print(2 * arr)
# print("++++++++")
# print(2 * arr - 1)







# q = np.array([[0.4], [0.6]])

# result = np.empty([5, 1])
# np.dot(status_score, q, result)
# print(result)
# print(np.dot(status_score, q))








# arr = np.random.normal(1.75, 0.1, (4, 5))
# print(arr)
#
# arr_m = arr[1:2, 2:4]
# print(arr_m)

# status_score = np.array([[[80, 88], [82, 81], [84, 75], [86, 83], [75, 81]],
#                          [[81, 88], [83, 81], [85, 75], [87, 83], [76, 81]]])
# print(status_score > 80)
# modify = np.where(status_score >= 80, "Good", "Bad")
# print(modify)
# print(np.amax(status_score, axis=2))
# print(np.mean(status_score, axis=2))
# print(np.std(status_score, axis=2))

# status_score[0, 1:3, 0] = status_score[0, 1:3, 0] + 5
# print(status_score)

