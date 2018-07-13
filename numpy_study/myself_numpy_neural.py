import numpy as np
import time


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))

        
def logistic_derivative(x):
    return x * (1 - x)
    # return logistic(x) * (1 - logistic(x))


class NumpyNeuralNet:

    def __init__(self, layer, activation="logistic", isDbg=False):
        self.isDbg = isDbg

        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        self.layer_cnt = len(layer) - 1
        for cnt in range(self.layer_cnt):
            self.weights.append(2 * np.random.random((layer[cnt], layer[cnt + 1])) - 1)
            if self.isDbg:
                print("Initialize weights:")
                print(np.shape(self.weights[cnt]))
                print(self.weights[cnt])

    def run(self, in_x, in_y):
        net_fp = []
        net_bp = []

        # Forward propagation
        net_fp.append(self.activation(np.dot(in_x, self.weights[0])))
        if self.layer_cnt > 1:
            for cnt in range(1, self.layer_cnt):
                net_fp.append(self.activation(np.dot(net_fp[cnt - 1], self.weights[cnt])))
                if self.isDbg:
                    print("Forward propagation: ", cnt)
                    print(np.shape(net_fp[cnt]))

        # Backward propagation
        net_bp.append((in_y - net_fp[-1]) * self.activation_deriv(net_fp[-1]))
        if self.layer_cnt > 1:
            for cnt in range(1, self.layer_cnt):
                net_bp.append(np.dot(net_bp[cnt - 1], self.weights[-cnt].T)
                                   * self.activation_deriv(net_fp[-1 - cnt]))
                if self.isDbg:
                    print("Backward propagation: ", cnt)
                    print(np.shape(net_bp[cnt]))

        # Update weights
        for cnt in range(1, self.layer_cnt):
            self.weights[self.layer_cnt - cnt] += np.dot(net_fp[-1 - cnt].T, net_bp[cnt - 1])
            if self.isDbg:
                print("Update weights: ", self.layer_cnt - cnt)
                print(np.shape(self.weights[self.layer_cnt - cnt]))
        self.weights[0] += np.dot(in_x.T, net_bp[-1])

    def test(self, test_x):
        net_fp = []

        # Forward propagation
        net_fp.append(self.activation(np.dot(test_x, self.weights[0])))
        if self.layer_cnt > 1:
            for cnt in range(1, self.layer_cnt):
                net_fp.append(self.activation(np.dot(net_fp[cnt - 1], self.weights[cnt])))
                if self.isDbg:
                    print("Test forward propagation: ", cnt)
                    print(np.shape(net_fp[cnt]))

        return net_fp[-1]


isDbg = False
if __name__ == '__main__':
    net = NumpyNeuralNet([2, 8, 1], activation="logistic", isDbg=isDbg)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0, 1, 1, 0]]).T
    test_x = np.array([[1, 0], [1, 1]])
    for i in range(100000):
        net.run(x, y)
        if i % 100 == 0:
            print(net.test(test_x))
        if isDbg:
            time.sleep(1)