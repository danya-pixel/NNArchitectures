from turtle import st
import numpy as np
from utils import Param, col2im, im2col


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

    def params(self):
        pass


class ReLULayer(Layer):
    def __init__(self):
        self.X = None
        self.isUpdatable = False

    def forward(self, X):
        self.X = X
        return X * (X > 0)

    def backward(self, d_out):
        d_result = (self.X > 0) * d_out
        return d_result

    def params(self):
        return {}


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
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None
        self.isUpdatable = True

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis = 0, keepdims = True)
        d_result = np.dot(d_out, self.W.value.T)
        return d_result

    def params(self):
        return { 'W': self.W, 'B': self.B }


class ConvolutionalLayer(Layer):
    def __init__(self, in_channels, out_channels,
                 filter_size, padding, stride=1):
        '''
        Initializes the layer
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )
        self.B = Param(np.zeros(out_channels))
        
        self.X = None
        self.isUpdatable = True

    def forward(self, X):
        batch_size, height, width, _ = X.shape

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1

        self.X = np.pad(X, ((0, 0), (self.padding, self.padding),
                        (self.padding, self.padding), (0, 0)), 'constant', constant_values=0)
        W = self.W.value.reshape(
            self.filter_size * self.filter_size * self.in_channels, self.out_channels)

        out_shape = (batch_size, out_height, out_width, self.out_channels)
        output = np.zeros(out_shape)

        for y in range(out_height):
            for x in range(out_width):
                h_end, w_end = y + self.filter_size, x + self.filter_size

                I = self.X[:, y:h_end, x:w_end, :].reshape(batch_size, -1)
                output[:, y, x, :] = np.dot(I, W)

        return output + self.B.value

    def backward(self, d_out):
        batch_size, height, width, _ = self.X.shape
        _, out_height, out_width, _ = d_out.shape

        dX = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                h_end, w_end = y + self.filter_size, x + self.filter_size

                dX[:, y:h_end, x:w_end, :] += np.dot(d_out[:, y, x, :], self.W.value.reshape(
                    -1, self.out_channels).T).reshape(batch_size, -1)

                self.W.grad += np.dot(self.X[:, y:h_end, x:w_end, :].reshape(
                    batch_size, -1).T, d_out[:, y, x, :]).reshape(self.W.value.shape)

        self.B.grad = np.sum(d_out, axis=(0, 1, 2))
        return dX[:, self.padding: (height - self.padding), self.padding: (width - self.padding), :]

    def params(self):
        return {'W': self.W, 'B': self.B}



class Conv2D():
    
    def __init__(self, nb_filters, filter_size, nb_channels, stride=1, padding=0):
        self.n_F = nb_filters
        self.f = filter_size
        self.n_C = nb_channels
        self.s = stride
        self.p = padding
        self.isUpdatable = True

        # Xavier-Glorot initialization - used for sigmoid, tanh.
        self.W = Param(np.random.randn(self.n_F, self.n_C, self.f, self.f) * np.sqrt(1. / (self.f)))
        self.b = Param(np.random.randn(self.n_F) * np.sqrt(1. / self.n_F))
     
        self.cache = None

    def forward(self, X):
        """
            Performs a forward convolution.
           
            Parameters:
            - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
            Returns:
            - out: previous layer convolved.
        """
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = self.n_F
        n_H = int((n_H_prev + 2 * self.p - self.f)/ self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f)/ self.s) + 1
        
        X_col = im2col(X, self.f, self.f, self.s, self.p)
        w_col = self.W.value.reshape((self.n_F, -1))
        b_col = self.b.value.reshape(-1, 1)
        # Perform matrix multiplication.
        out = w_col @ X_col + b_col
        # Reshape back matrix to image.
        out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_H, n_W))
        self.cache = X, X_col, w_col
        return out

    def backward(self, dout):
        """
            Distributes error from previous layer to convolutional layer and
            compute error for the current convolutional layer.
            Parameters:
            - dout: error from previous layer.
            
            Returns:
            - dX: error of the current convolutional layer.
            - self.W['grad']: weights gradient.
            - self.b['grad']: bias gradient.
        """
        X, X_col, w_col = self.cache
        m, _, _, _ = X.shape
        # Compute bias gradient.
        self.b['grad'] = np.sum(dout, axis=(0,2,3))
        # Reshape dout properly.
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, m))
        dout = np.concatenate(dout, axis=-1)
        # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
        dX_col = w_col.T @ dout
        # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
        dw_col = dout @ X_col.T
        # Reshape back to image (col2im).
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        # Reshape dw_col into dw.
        self.W.grad = dw_col.reshape((dw_col.shape[0], self.n_C, self.f, self.f))
                
        return dX

class MaxPoolingLayer(Layer):
    def __init__(self, pool_size, stride):
        '''
        Initializes the Max Pool layer
        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.masks = {}
        self.isUpdatable = False

    def forward(self, X):
        batch_size, channels, height, width,  = X.shape
        self.X = X
        self.masks.clear()
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        output_shape = (batch_size, out_height, out_width, channels)
        output = np.zeros(output_shape)

        for y in range(out_height):
            for x in range(out_width):
                h_begin, w_begin = y * self.stride, x * self.stride
                h_end, w_end = h_begin + self.pool_size, w_begin + self.pool_size

                I = X[:, :, h_begin:h_end, w_begin:w_end]
                self.build_mask(x=I, pos=(y, x))
                output[:, y, x, :] = np.max(I, axis=(1, 2))

        return output

    def backward(self, d_out):
        _, out_height, out_width, _ = d_out.shape
        dX = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):
                h_begin, w_begin = y * self.stride, x * self.stride
                h_end, w_end = h_begin + self.pool_size, w_begin + self.pool_size

                dX[:, :, h_begin:h_end, w_begin:w_end] += d_out[:,:,
                                                                y:y + 1, x:x + 1] * self.masks[(y, x)]
        return dX

    def build_mask(self, x, pos):
        mask = np.zeros_like(x)
        n, c, h, w = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, c, h * w)[n_idx, idx, c_idx] = 1
        self.masks[pos] = mask

    def params(self):
        return {}

class Flattener:
    def __init__(self):
        self.X_shape = None
        self.isUpdatable = False

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        return {}

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    preds_copy = np.copy(predictions)
    preds_copy = (preds_copy.T - np.max(preds_copy, axis = 1)).T
    exp_pred = np.exp(preds_copy)
    exp_sum = np.sum(exp_pred, axis = 1)
    probs = (exp_pred.T / exp_sum).T
    
    batch_size = predictions.shape[0]
    loss = np.sum(-np.log(probs[range(batch_size), target_index])) / batch_size
    
    dprediction = probs
    dprediction[range(batch_size), target_index] -= 1
    dprediction /= batch_size
    return loss, dprediction


class MaxPool():
    
    def __init__(self, filter_size, stride=1, padding=0):
        self.f = filter_size
        self.s = stride
        self.p = padding
        self.cache = None

    def forward(self, X):
        """
            Apply average pooling.
            Parameters:
            - X: Output of activation function.
            
            Returns:
            - A_pool: X after average pooling layer. 
        """
        self.cache = X

        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.p - self.f)/ self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f)/ self.s) + 1
        
        X_col = im2col(X, self.f, self.f, self.s, self.p)
        X_col = X_col.reshape(n_C, X_col.shape[0]//n_C, -1)
        A_pool = np.max(X_col, axis=1)
        # Reshape A_pool properly.
        A_pool = np.array(np.hsplit(A_pool, m))
        A_pool = A_pool.reshape(m, n_C, n_H, n_W)

        return A_pool

    def backward(self, dout):
        """
            Distributes error through pooling layer.
            Parameters:
            - dout: Previous layer with the error.
            
            Returns:
            - dX: Conv layer updated with error.
        """
        X = self.cache
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.p - self.f)/ self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f)/ self.s) + 1

        dout_flatten = dout.reshape(n_C, -1) / (self.f * self.f)
        dX_col = np.repeat(dout_flatten, self.f*self.f, axis=0)
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        # Reshape dX properly.
        dX = dX.reshape(m, -1)
        dX = np.array(np.hsplit(dX, n_C_prev))
        dX = dX.reshape(m, n_C_prev, n_H_prev, n_W_prev)
        return dX
