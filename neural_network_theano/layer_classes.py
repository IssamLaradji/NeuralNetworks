# Authors: Issam H. Laradji <issam.laradji@gmail.com>
# Licence: BSD 3 clause

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import numpy as np
import base as ut

ACTIVATION_FUNCTIONS = {"logistic" : T.nnet.sigmoid, "tanh": T.tanh}

### LAYER CLASSES

class fully_connected_layer():
    def __init__(self, n_hidden=50, activation="tanh"):
        self.n_hidden = n_hidden
        self.activation = activation

    def construct(self, X, n_input):
        W, b = ut.init_weights(n_input, self.n_hidden)

        self.output = T.dot(X, W) + b

        if self.activation:
            self.output = ACTIVATION_FUNCTIONS[self.activation](self.output)

        self.n_outputs = self.n_hidden
        self.params = [W, b]


class convolutional_layer():
    def __init__(self, n_kernels=10, filter_shape=(5, 5), pool_size=(2, 2), activation="tanh"):
        self.activation = activation
        self.single_filter_shape = filter_shape
        self.n_kernels = n_kernels
        self.pool_size = pool_size

    def construct(self, X, n_input_kernels, image_shape):

        self.filter_shape = tuple(list([self.n_kernels]) + list([n_input_kernels]) \
                                  + list(self.single_filter_shape))

        W, b = ut.init_weights_conv(self.filter_shape)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=X,
            filters=W,
            filter_shape=self.filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=self.pool_size,
            ignore_border=True
        )
        self.output = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))

        if self.activation:
            self.output = ACTIVATION_FUNCTIONS[self.activation](self.output)

        self.n_outputs = self.n_kernels * 4
        self.params = [W, b]

### LOSS FUNCTIONS

class square_loss():
    def __init__(self):
        pass

    def construct(self, X, y, n_input, n_output):
        W, b = ut.init_weights(n_input, n_output)

        y_pred = T.dot(X, W) + b

        self.y_pred = y_pred
        self.loss_function = 0.5 * T.sum((y_pred - y)**2)
        self.params = [W, b]

class logistic_loss():
    def __init__(self):
        pass

    def construct(self, X, y, n_input, n_output):
        W, b = ut.init_weights(n_input, n_output)

        y_pred = T.nnet.softmax(T.dot(X, W) + b)

        self.y_pred = y_pred
        self.loss_function = - T.sum(y * T.log(y_pred))
        self.params = [W, b]