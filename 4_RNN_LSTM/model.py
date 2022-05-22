import numpy as np
from layers import FullyConnected, RNN_block, LSTM_block

from metrics import multiclass_accuracy
from tqdm import tqdm


class RNN:
    def __init__(self, features_num=12, hidden_rnn_size=10):
        self.layers = []

        self.layers.append(
            RNN_block(input_size=features_num, hidden_size=hidden_rnn_size)
        )
        self.layers.append(FullyConnected(hidden_rnn_size, 1))

    def forward(self, X):

        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict_accuracy(self, X, y, batch_size):
        self.training_mode[0] = False
        indices = np.arange(X.shape[0])

        sections = np.arange(batch_size, X.shape[0], batch_size)
        batches_indices = np.array_split(indices, sections)

        pred = np.zeros_like(y)
        probs = np.zeros((y.shape[0], 10))

        for batch_indices in tqdm(batches_indices):
            batch_X = X[batch_indices]
            batch_X_gpu = np.asarray(batch_X)

            out_batch = self.forward(batch_X_gpu).get()
            pred_batch = np.argmax(out_batch, axis=1)
            pred[batch_indices] = pred_batch

        return multiclass_accuracy(pred, y)

    def params(self):
        result = {}
        for layer_num, layer in enumerate(self.layers):
            for param_name, param in layer.params().items():
                result[f"{param_name} {layer.name}_{layer_num}"] = param

        return result


class LSTM:
    def __init__(self, features_num=12, hidden_rnn_size=10):
        self.layers = []

        self.layers.append(
            LSTM_block(input_size=features_num, hidden_size=hidden_rnn_size)
        )
        self.layers.append(FullyConnected(hidden_rnn_size, 1))

    def forward(self, X):

        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict_accuracy(self, X, y, batch_size):
        self.training_mode[0] = False
        indices = np.arange(X.shape[0])

        sections = np.arange(batch_size, X.shape[0], batch_size)
        batches_indices = np.array_split(indices, sections)

        pred = np.zeros_like(y)

        for batch_indices in tqdm(batches_indices):
            batch_X = X[batch_indices]
            batch_X_gpu = np.asarray(batch_X)

            out_batch = self.forward(batch_X_gpu).get()
            pred_batch = np.argmax(out_batch, axis=1)
            pred[batch_indices] = pred_batch

        return multiclass_accuracy(pred, y)

    def params(self):
        result = {}
        for layer_num, layer in enumerate(self.layers):
            for param_name, param in layer.params().items():
                result[f"{param_name} {layer.name}_{layer_num}"] = param

        return result
