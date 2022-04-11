import numpy as np
from layers import Layer


class Perceptron():
    def __init__(self, loss):
        self.layers_num = 0
        self.layers = []
        self.optimizer = None
        self.loss = loss

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)
        self.layers_num += 1

    def forward(self, X):
        l_out = self.layers[0].forward(X)

        for layer in self.layers[1:self.layers_num-2]:
            l_out = layer.forward(l_out)

        y_pred = self.layers[-2].forward(l_out)
        _ = self.layers[-1].forward(y_pred)

        return y_pred

    def backward(self, y_pred, y_train):
        loss_back = self.loss.derivative(y_pred, y_train)
        l_back = self.layers[-1].backward(loss_back)
        for layer in self.layers[self.layers_num-2:0:-1]:
            l_back = layer.backward(l_back)

    def update(self, learning_rate: float) -> None:
        if self.optimizer is not None:
            self.optimizer.step(self.layers, learning_rate)
        else:
            for layer in self.layers:
                if layer.isUpdatable:
                    layer.W.value -= learning_rate * layer.W.grad

    def get_weights(self) -> list:
        weights = []
        idx = 0
        for layer in self.layers:
            if layer.isUpdatable:
                weights.append({'idx': idx, 'w': layer.W.value})
            else:
                weights.append(None)
            idx += 1
        return weights

    def upload_weights(self, weights: list) -> None:
        for weight in weights:
            if weight is not None:
                self.layers[weight['idx']].W.value = weight['w']

    def fit(self, X_train, y_train, learning_rate: int, optimizer=None,
            reg=None, num_epochs: int = 50):
        best_loss = np.inf
        loss_history = []
        best_params = []
        self.optimizer = optimizer
        self.reg = reg

        for epoch in range(num_epochs):
            y_pred = self.forward(X_train)
            self.backward(y_pred, y_train)

            loss = self.loss.compute(y_pred, y_train)

            if self.reg is not None:
                loss = self.reg.regularize(self.layers, loss)

            if loss < best_loss:
                best_loss = loss
                best_params = self.get_weights()

            if epoch % 250 == 0:
                print(epoch, 'Loss: ', loss)

            self.update(learning_rate)
            loss_history.append(loss)

        return loss_history, best_params
