import numpy as np
from utils import Param


class Layer:
    """
    General layer class
    """
    def __init__(self):
        self.isUpdatable = None
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class SigmoidLayer(Layer):
    def __init__(self):
        self.X = None
        self.isUpdatable = False

    def sigmoid(self, X):
        return 1/(1 + np.exp(-X))

    def forward(self, X):
        self.X = X
        return self.sigmoid(X)

    def backward(self, d_out):
        d_result = d_out * (self.sigmoid(self.X) * (1 - self.sigmoid(self.X)))
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer(Layer):
    def __init__(self, n_input, n_output):
        self.W = Param(1 * np.random.randn(n_input, n_output))
        self.X = None
        self.isUpdatable = True

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.value)

    def backward(self, d_out):
        self.W.grad = np.dot(self.X.T, d_out)
        d_result = np.dot(d_out, self.W.value.T)
        return d_result

    def params(self):
        return {'W': self.W}
