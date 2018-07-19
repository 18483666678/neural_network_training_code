import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1, 1, 0]]).T
w0 = 2*np.random.random((2, 8)) - 1
w1 = 2*np.random.random((8, 1)) - 1
for i in range(100000):
    l1_out = 1/(1+np.exp(-(np.dot(x, w0))))
    l2_out = 1/(1+np.exp(-(np.dot(l1_out, w1))))
    l2_delta = (y - l2_out) * (l2_out * (1 - l2_out))
    l1_delta = np.dot(l2_delta, w1.T) * (l1_out * (1 - l1_out))
    w1 += np.dot(l1_out.T, l2_delta)
    w0 += np.dot(x.T, l1_delta)

    if i % 100 == 0:
        t_x = np.array([[0, 0], [0, 1]])
        t_l1_out = 1/(1+np.exp(-(np.dot(t_x, w0))))
        t_l2_out = 1/(1+np.exp(-(np.dot(t_l1_out, w1))))
        print(t_l2_out)