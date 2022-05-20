from utils import im2col, col2im
import numpy as np


class Conv:
    def __init__(self, nb_filters, filter_size, nb_channels, stride=1, padding=0):
        self.n_F = nb_filters
        self.f = filter_size
        self.n_C = nb_channels
        self.s = stride
        self.p = padding

        self.W = {
            "val": np.random.randn(self.n_F, self.n_C, self.f, self.f)
            * np.sqrt(1.0 / (self.f)),
            "grad": np.zeros((self.n_F, self.n_C, self.f, self.f)),
        }
        self.b = {
            "val": np.random.randn(self.n_F) * np.sqrt(1.0 / self.n_F),
            "grad": np.zeros((self.n_F)),
        }

        self.cache = None

    def forward(self, X):
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = self.n_F
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        X_col = im2col(X, self.f, self.f, self.s, self.p)
        w_col = self.W["val"].reshape((self.n_F, -1))
        b_col = self.b["val"].reshape(-1, 1)
        out = w_col @ X_col + b_col
        out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_H, n_W))
        self.cache = X, X_col, w_col
        return out

    def backward(self, dout):
        X, X_col, w_col = self.cache
        m, _, _, _ = X.shape
        self.b["grad"] = np.sum(dout, axis=(0, 2, 3))
        dout = dout.reshape(
            dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3]
        )
        dout = np.array(np.vsplit(dout, m))
        dout = np.concatenate(dout, axis=-1)
        dX_col = w_col.T @ dout
        dw_col = dout @ X_col.T
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        self.W["grad"] = dw_col.reshape((dw_col.shape[0], self.n_C, self.f, self.f))

        return dX, self.W["grad"], self.b["grad"]


class MaxPool:
    def __init__(self, filter_size, stride=1, padding=0):
        self.f = filter_size
        self.s = stride
        self.p = padding
        self.cache = None

    def forward(self, X):
        self.cache = X

        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        X_col = im2col(X, self.f, self.f, self.s, self.p)
        X_col = X_col.reshape(n_C, X_col.shape[0] // n_C, -1)
        A_pool = np.max(X_col, axis=1)
        A_pool = np.array(np.hsplit(A_pool, m))
        A_pool = A_pool.reshape(m, n_C, n_H, n_W)

        return A_pool

    def backward(self, dout):
        X = self.cache
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = n_C_prev

        dout_flatten = dout.reshape(n_C, -1) / (self.f * self.f)
        dX_col = np.repeat(dout_flatten, self.f * self.f, axis=0)
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        dX = dX.reshape(m, -1)
        dX = np.array(np.hsplit(dX, n_C_prev))
        dX = dX.reshape(m, n_C_prev, n_H_prev, n_W_prev)
        return dX


class Fc:
    def __init__(self, row, column):
        self.row = row
        self.col = column

        self.W = {
            "val": np.random.randn(self.row, self.col) * np.sqrt(1.0 / self.col),
            "grad": 0,
        }
        self.b = {
            "val": np.random.randn(1, self.row) * np.sqrt(1.0 / self.row),
            "grad": 0,
        }

        self.cache = None

    def forward(self, fc):
        self.cache = fc
        A_fc = np.dot(fc, self.W["val"].T) + self.b["val"]
        return A_fc

    def backward(self, deltaL):
        fc = self.cache
        m = fc.shape[0]

        self.W["grad"] = (1 / m) * np.dot(deltaL.T, fc)
        self.b["grad"] = (1 / m) * np.sum(deltaL, axis=0)

        new_deltaL = np.dot(deltaL, self.W["val"])
        return new_deltaL, self.W["grad"], self.b["grad"]


class AdamGD:
    def __init__(self, lr, beta1, beta2, epsilon, params):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.params = params

        self.momentum = {}
        self.rmsprop = {}

        for key in self.params:
            self.momentum["vd" + key] = np.zeros(self.params[key].shape)
            self.rmsprop["sd" + key] = np.zeros(self.params[key].shape)

    def update_params(self, grads):

        for key in self.params:
            self.momentum["vd" + key] = (self.beta1 * self.momentum["vd" + key]) + (
                1 - self.beta1
            ) * grads["d" + key]
            self.rmsprop["sd" + key] = (self.beta2 * self.rmsprop["sd" + key]) + (
                1 - self.beta2
            ) * (grads["d" + key] ** 2)
            self.params[key] = self.params[key] - (
                self.lr * self.momentum["vd" + key]
            ) / (np.sqrt(self.rmsprop["sd" + key]) + self.epsilon)

        return self.params


class Sigmoid:
    def __init__(self):
        self.cache = None
        self.out = None

    def forward(self, X):
        self.cache = X
        self.out = 1 / (1 + np.exp(-X))
        return self.out

    def backward(self, new_deltaL):
        return new_deltaL * self.out * (1 - self.out)


class Softmax:
    def __init__(self):
        pass

    def forward(self, X):
        e_x = np.exp(X - np.max(X))
        return e_x / np.sum(e_x, axis=1)[:, np.newaxis]

    def backward(self, y_pred, y):
        return y_pred - y


class CrossEntropyLoss:
    def __init__(self):
        pass

    def get(self, y_pred, y):
        loss = -np.sum(y * np.log(y_pred))
        return loss
