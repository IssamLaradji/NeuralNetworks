import theano
import theano.tensor as T
import numpy as np


def init_weights(n_input, n_output):
    W_ = np.asarray(np.random.rand(n_input, n_output), dtype=theano.config.floatX)
    b_ = np.asarray(np.random.rand(n_output), dtype=theano.config.floatX)

    W = theano.shared(value=W_, borrow=True)
    b = theano.shared(value=b_, borrow=True)

    return W, b

def init_weights_conv(filter_shape):
    W_ = np.asarray(np.random.random(filter_shape), dtype=theano.config.floatX)
    b_ = np.asarray(np.random.random((filter_shape[0],)), dtype=theano.config.floatX)

    W = theano.shared(value=W_, borrow=True)
    b = theano.shared(value=b_, borrow=True)

    return W, b