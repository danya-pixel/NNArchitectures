import numpy as np


class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


def l2_regularization(W, reg_strength):
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax(predictions):
    if len(predictions.shape) == 2:
        s = np.max(predictions, axis=1)
        e_x = np.exp(predictions - s[:, np.newaxis])
        div = np.sum(e_x, axis=1)
        return e_x / div[:, np.newaxis]
    else:
        exps = np.exp(predictions - np.max(predictions))
        return exps / np.sum(exps)


def cross_entropy_loss(probs, target_index):
    if type(target_index) == int:
        target_index = np.asarray([target_index])
    if len(target_index.shape) == 1:
        probs = probs[target_index]
    else:
        probs = probs[range(probs.shape[0]), target_index[:, 0]]

    probs = np.clip(probs, 0.001, 1)

    return -np.mean(np.log(probs))


def softmax_with_cross_entropy(preds, target_index):
    if type(target_index) == int:
        index = np.asarray([target_index])
    else:
        index = target_index.copy()

    if index.ndim == 1 and index.size > 1:
        index = index.reshape(-1, 1)

    prob = softmax(preds.copy())

    loss = cross_entropy_loss(prob, index)

    y = np.zeros_like(preds)

    if len(index.shape) == 1:
        y[index] = 1
    else:
        y[range(y.shape[0]), index[:, 0]] = 1

    dprediction = prob - 1 * y

    if preds.ndim == 2:
        dprediction = dprediction / preds.shape[0]
    return loss, dprediction


def MSELoss(pred, y):
    return np.mean((pred - y) ** 2)


class ReLU:
    def __init__(self):
        self.name = "relu"
        self.indexes = None
        pass

    def forward(self, X):
        if self.indexes is None or self.indexes.shape[0] != X.shape[0]:
            self.indexes = np.zeros_like(X, dtype=np.bool)

        np.less(X, 0, out=self.indexes)

        result = X
        np.multiply(result, self.indexes, out=result)
        return result

    def backward(self, d_out):
        d_result = d_out
        np.multiply(d_result, self.indexes, out=d_result)
        return d_result

    def params(self):
        return {}


class FullyConnected:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None
        self.name = "FC"

    def forward(self, X):

        self.X = X
        res = np.dot(X, self.W.value) + self.B.value

        return res

    def backward(self, d_out):
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.array([np.sum(d_out, axis=0)])
        gradX = np.dot(d_out, self.W.value.T)
        return gradX

    def params(self):
        return {"W": self.W, "B": self.B}


class Flattener:
    def __init__(self):
        self.X_shape = None
        self.name = "flattener"

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X_shape = batch_size, height, width, channels
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):

        return {}


class Tanh:
    def __init__(self):
        self.name = "tanh"

    def forward(self, X):
        return np.tanh(X)

    def backward(self, d_out):
        return 1 - d_out**2

    def params(self):
        return {}


class Sigmoid:
    def __init__(self):
        self.name = "sigmoid"

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def backward(self, d_out):
        return d_out * (1 - d_out)

    def params(self):
        return {}


class RNN_block:
    def __init__(self, input_size, hidden_size):
        self.name = "RNN"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ax = Param(0.001 * np.random.randn(input_size, hidden_size))
        self.W_aa = Param(0.001 * np.random.randn(hidden_size, hidden_size))
        self.B = Param(0.001 * np.random.randn(1, hidden_size))

    def forward(self, input_X):
        batch_size = input_X.shape[0]
        self.input_X = np.swapaxes(input_X, 0, 1)

        self.relus = [ReLU() for x in input_X]
        hidden = np.zeros((batch_size, self.hidden_size))

        self.hidden_list = [hidden]
        self.y_preds = []

        for input_x_t, relu in zip(self.input_X, self.relus):
            input_tanh = (
                np.dot(input_x_t, self.W_ax.value)
                + np.dot(hidden, self.W_aa.value)
                + self.B.value
            )

            hidden = relu.forward(input_tanh)

            if np.any(np.isnan(hidden)):
                return None

            self.hidden_list.append(hidden)

        return hidden

    def backward(self, d_out):

        for input_x_t, hidden, relu in reversed(
            list(zip(self.input_X, self.hidden_list[:-1], self.relus))
        ):
            dtanh = relu.backward(d_out)
            self.B.grad += np.array([np.sum(d_out, axis=0)])
            self.W_ax.grad += np.dot(input_x_t.T, dtanh)
            self.W_aa.grad += np.dot(hidden.T, dtanh)
            d_out = np.dot(dtanh, self.W_aa.value.T)

        return None

    def params(self):
        return {"W_ax": self.W_ax, "W_aa": self.W_aa, "B": self.B}


class LSTM_block:
    def __init__(self, input_size, hidden_size):
        self.name = "LSTM"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_f = Param(0.001 * np.random.randn(input_size + hidden_size, hidden_size))
        self.W_i = Param(0.001 * np.random.randn(input_size + hidden_size, hidden_size))
        self.W_c = Param(0.001 * np.random.randn(input_size + hidden_size, hidden_size))
        self.W_o = Param(0.001 * np.random.randn(input_size + hidden_size, hidden_size))

        self.B_f = Param(0.001 * np.random.randn(1, hidden_size))
        self.B_i = Param(0.001 * np.random.randn(1, hidden_size))
        self.B_c = Param(0.001 * np.random.randn(1, hidden_size))
        self.B_o = Param(0.001 * np.random.randn(1, hidden_size))

        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def forward(self, input_X):
        batch_size = input_X.shape[0]
        self.input_X = np.swapaxes(input_X, 0, 1)

        hidden = np.zeros((batch_size, self.hidden_size))
        cell_state = np.zeros((batch_size, self.hidden_size))

        self.hidden_list = [hidden]
        self.cell_state_list = [cell_state]

        self.y_preds = []
        self.o_t_list = []
        self.i_t_list = []
        self.f_t_list = []
        self.C_t_wave_list = []

        for input_x_t in self.input_X:
            h_x_concat = np.concatenate((hidden, input_x_t), axis=1)

            f_t = self.sigmoid.forward(h_x_concat @ self.W_f.value + self.B_f.value)
            i_t = self.sigmoid.forward(h_x_concat @ self.W_i.value + self.B_i.value)

            C_t_wave = self.tanh.forward(h_x_concat @ self.W_c.value + self.B_c.value)
            cell_state = f_t * cell_state + i_t * C_t_wave

            o_t = self.sigmoid.forward(h_x_concat @ self.W_o.value + self.B_o.value)
            hidden = o_t * self.tanh.forward(cell_state)

            self.hidden_list.append(hidden)
            self.cell_state_list.append(cell_state)
            self.o_t_list.append(o_t)
            self.i_t_list.append(i_t)
            self.f_t_list.append(f_t)
            self.C_t_wave_list.append(C_t_wave)

        return hidden

    def backward(self, d_out):
        d_cell_state = np.zeros_like(self.cell_state_list[0])
        for (
            input_x_t,
            hidden,
            cell_state,
            o_t,
            i_t,
            f_t,
            C_t_wave,
            prev_cell_state,
        ) in reversed(
            list(
                zip(
                    self.input_X,
                    self.hidden_list[:-1],
                    self.cell_state_list,
                    self.o_t_list,
                    self.i_t_list,
                    self.f_t_list,
                    self.C_t_wave_list,
                    self.cell_state_list[:-1],
                )
            )
        ):

            d_o_t = self.tanh.forward(cell_state) * d_out
            d_C_t = d_cell_state + d_out * o_t * (
                1 - self.tanh.forward(cell_state) ** 2
            )
            d_C_t_wave = d_C_t * i_t
            d_i_t = d_C_t * C_t_wave
            d_f_t = d_C_t * prev_cell_state

            d_f_t = f_t * (1 - f_t) * d_f_t
            d_i_t = i_t * (1 - i_t) * d_i_t

            d_o_t = o_t * (1 - o_t) * d_o_t
            d_z_t = (
                self.W_f.value @ d_f_t.T
                + self.W_i.value @ d_i_t.T
                + self.W_c.value @ d_C_t_wave.T
                + self.W_o.value @ d_o_t.T
            )

            d_out = d_z_t.T[: d_out.shape[0], : d_out.shape[1]]

            z = np.concatenate((hidden, input_x_t), axis=1)

            self.W_f.grad += z.T @ d_f_t
            self.B_f.grad += d_f_t.sum(axis=0).reshape(1, -1)

            self.W_i.grad += z.T @ d_i_t
            self.B_i.grad += d_i_t.sum(axis=0).reshape(1, -1)

            self.W_c.grad += z.T @ d_C_t_wave
            self.B_c.grad += d_C_t_wave.sum(axis=0).reshape(1, -1)

            self.W_o.grad += z.T @ d_o_t
            self.B_o.grad += d_o_t.sum(axis=0).reshape(1, -1)
        return None

    def params(self):
        return {
            "W_f": self.W_f,
            "W_i": self.W_i,
            "W_c": self.W_c,
            "W_o": self.W_o,
            "B_f": self.B_f,
            "B_i": self.B_i,
            "B_c": self.B_c,
            "B_o": self.B_o,
        }
