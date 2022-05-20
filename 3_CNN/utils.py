import os
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import concurrent.futures as cf
import idx2numpy


def load_mnist(folder):
    train_x = np.expand_dims(
        idx2numpy.convert_from_file(os.path.join(folder, "train-images-idx3-ubyte")),
        axis=3,
    )
    train_y = idx2numpy.convert_from_file(
        os.path.join(folder, "train-labels-idx1-ubyte")
    )
    test_x = np.expand_dims(
        idx2numpy.convert_from_file(os.path.join(folder, "t10k-images-idx3-ubyte")),
        axis=3,
    )
    test_y = idx2numpy.convert_from_file(os.path.join(folder, "t10k-labels-idx1-ubyte"))

    return train_x, train_y, test_x, test_y


def resize_dataset(dataset):
    args = [dataset[i : i + 1000] for i in range(0, len(dataset), 1000)]

    def f(chunk):
        return transform.resize(chunk, (chunk.shape[0], 1, 32, 32))

    with cf.ThreadPoolExecutor() as executor:
        res = executor.map(f, args)

    res = np.array([*res])
    res = res.reshape(-1, 1, 32, 32)
    return res


def dataloader(X, y, BATCH_SIZE):
    n = len(X)
    for t in range(0, n, BATCH_SIZE):
        yield X[t : t + BATCH_SIZE, ...], y[t : t + BATCH_SIZE, ...]


class Dataset:
    def __init__(self, trainx, trainy, valx, valy):
        self.trainx = trainx
        self.trainy = trainy
        self.valx = valx
        self.valy = valy
        self.train_len = len(trainx)
        self.val_len = len(valx)

    def data_loader(self, type, BATCH_SIZE=64):

        if type == "train":
            for t in range(0, self.train_len, BATCH_SIZE):
                yield self.trainx[t : t + BATCH_SIZE, ...], self.trainy[
                    t : t + BATCH_SIZE, ...
                ]

        if type == "val":
            for t in range(0, self.val_len, BATCH_SIZE):
                yield self.valx[t : t + BATCH_SIZE, ...], self.valy[
                    t : t + BATCH_SIZE, ...
                ]


def one_hot_encoding(y):
    N = y.shape[0]
    Z = np.zeros((N, 10))
    Z[np.arange(N), y] = 1
    return Z


def show_batch(train_X, train_y):
    num_row = 2
    num_col = 5
    _, axes = plt.subplots(num_row, num_col, figsize=(2 * num_col, 2 * num_row))
    for i in range(10):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(train_X[i][0], cmap="gray")
        ax.set_title("Label: {}".format(np.argmax(train_y[i])))
    plt.tight_layout()
    plt.show()


def prepare_for_neural_network(train_X, test_X):
    train_X = train_X.astype(np.float64) / 255.0
    test_X = test_X.astype(np.float64) / 255.0

    mean_image = np.mean(train_X, axis=0)
    train_X -= mean_image
    test_X -= mean_image

    return train_X, test_X


def random_split_train_val(X, y, percent_val, seed=42):
    np.random.seed(seed)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    num_val = int(X.shape[0] * percent_val)
    train_indices = indices[:-num_val]
    train_X = X[train_indices]
    train_y = y[train_indices]

    val_indices = indices[-num_val:]
    val_X = X[val_indices]
    val_y = y[val_indices]

    return train_X, train_y, val_X, val_y


def get_indices(X_shape, HF, WF, stride, pad):
    m, n_C, n_H, n_W = X_shape

    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    level1 = np.repeat(np.arange(HF), WF)
    level1 = np.tile(level1, n_C)
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    slide1 = np.tile(np.arange(WF), HF)
    slide1 = np.tile(slide1, n_C)
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d


def im2col(X, HF, WF, stride, pad):
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols


def col2im(dX_col, X_shape, HF, WF, stride, pad):
    N, D, H, W = X_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))

    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]
