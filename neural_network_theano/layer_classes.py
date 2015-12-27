# Authors: Issam H. Laradji <issam.laradji@gmail.com>
# Licence: BSD 3 clause

import theano
import theano.tensor as T
import numpy as np

ACTIVATION_FUNCTIONS = {"logistic" : T.nnet.sigmoid, "tanh": T.tanh}

def init_weights(n_input, n_output):
    W_ = np.asarray(np.random.rand(n_input, n_output), dtype=theano.config.floatX)
    b_ = np.asarray(np.random.rand(n_output), dtype=theano.config.floatX)

    W = theano.shared(value=W_, borrow=True)
    b = theano.shared(value=b_, borrow=True)

    return W, b

class fully_connected_layer():
    def __init__(self, n_hidden=50, activation="logistic"):
        self.n_hidden = n_hidden
        self.activation = activation

    def construct(self, X, n_input):
        W, b = init_weights(n_input, self.n_hidden)

        self.output = T.dot(X, W) + b

        if self.activation:
            self.output = ACTIVATION_FUNCTIONS[self.activation](self.output)

        self.n_output = self.n_hidden
        self.params = [W, b]

class square_loss():
    def __init__(self):
        pass

    def construct(self, X, y, n_input, n_output):
        W, b = init_weights(n_input, n_output)

        y_pred = T.dot(X, W) + b

        self.y_pred = y_pred
        self.loss_function = 0.5 * T.sum((y_pred - y)**2)
        self.params = [W, b]

class logistic_loss():
    def __init__(self):
        pass

    def construct(self, X, y, n_input, n_output):
        W, b = init_weights(n_input, n_output)

        y_pred = T.nnet.softmax(T.dot(X, W) + b)

        self.y_pred = y_pred
        self.loss_function = - T.sum(y * T.log(y_pred))
        self.params = [W, b]