import numpy as np


class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class MSE():
    def __init__(self):
        pass

    def derivative(self, pred, y) -> float:
        return pred-y

    def compute(self, pred, y) -> float:
        return np.sum(pred-y)**2
