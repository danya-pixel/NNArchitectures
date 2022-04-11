import numpy as np
from layers import Layer


class Adam():
    def __init__(self, model, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.t = 1
        self.v = [0] * len(model.layers)
        self.s = [0] * len(model.layers)

    def step(self, layers: list[Layer], learning_rate):
        for i, layer in enumerate(layers):
            if layer.isUpdatable:
                self.v[i] = self.beta_1 * self.v[i] + \
                    (1 - self.beta_1) * layer.W.grad
                self.s[i] = self.beta_2 * self.s[i] + \
                    (1 - self.beta_2) * np.square(layer.W.grad)

                v_bias_corr = self.v[i] / (1 - self.beta_1 ** self.t)
                s_bias_corr = self.s[i] / (1 - self.beta_2 ** self.t)
                layer.W.value -= learning_rate * v_bias_corr / \
                    (np.sqrt(s_bias_corr) + self.epsilon)
        self.t += 1


class L2():
    def __init__(self, reg_strength):
        self.reg_strength = reg_strength

    def regularize(self, layers: list, loss: float) -> float:
        for layer in layers:
            if layer.isUpdatable:
                loss += self.reg_strength * np.sum(layer.W.value)
                layer.W.grad += 2 * self.reg_strength * layer.W.value
        return loss
