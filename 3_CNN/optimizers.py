import numpy as np
from copy import deepcopy


class Adam:
    def __init__(self, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = np.array(epsilon).astype(np.float32)

        self.t = 1
        self.v = 0
        self.s = 0

        self.sqrt_res = None

    def update(self, weights, grad, learning_rate):
        self.v = self.beta_1 * self.v + (1 - self.beta_1) * grad
        self.s = self.beta_2 * self.s + (1 - self.beta_2) * np.square(grad)

        v_bias_corr = self.v / (1 - self.beta_1**self.t)
        s_bias_corr = self.s / (1 - self.beta_2**self.t)

        weights -= learning_rate * v_bias_corr / (np.sqrt(s_bias_corr) + self.epsilon)
        self.t += 1


def setup_optimizers(model):
    params = model.params()
    model.optimizers = {}
    for param_name, param in params.items():
        model.optimizers[param_name] = deepcopy(model.optim)


class MomentumSGD:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity = 0

    def update(self, w, d_w, learning_rate):
        self.velocity = self.momentum * self.velocity - learning_rate * d_w
        w = w + self.velocity
