"""Neural Network
"""

# Authors: Issam H. Laradji <issam.laradji@gmail.com>
# Licence: BSD 3 clause

import time
from abc import ABCMeta, abstractmethod

import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.externals import six
from sklearn.preprocessing import LabelBinarizer

from neural_network_theano.layer_classes import fully_connected_layer, convolutional_layer
from neural_network_theano.layer_classes import square_loss, logistic_loss
from update_rules import sgd_class


class BaseNeuralNetwork(six.with_metaclass(ABCMeta, BaseEstimator)):
    @abstractmethod
    def __init__(self, layers, loss_function, learning_rate, batch_size, update_algorithm, max_epcohs, verbose):
        self.layers = layers
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epcohs
        self.update_algorithm = update_algorithm
        self.verbose = verbose

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        A = T.matrix('A')
        b = T.matrix('b')

        self.param_list = []

        output = A
        n_output = n_features

        # 1. Construct layers
        for layer in self.layers:
            if isinstance(layer, fully_connected_layer):
                layer.construct(X=output, n_input=n_output)

                output = layer.output
                n_output = layer.n_outputs
                self.param_list += layer.params

            elif isinstance(layer, convolutional_layer):
                n_dim = int(np.sqrt(X.shape[1]))
                assert n_dim**2 == X.shape[1]
                image_shape = (self.batch_size, 1, n_dim, n_dim)
                output = A.reshape(image_shape)

                layer.construct(X=output, n_input_kernels=1, image_shape=image_shape)

                output = layer.output
                output = output.flatten(2)
                n_output = layer.n_outputs
                self.param_list += layer.params

            else:
                raise("Class not implemented!")

        # 2. Set loss function
        if self.loss_function == "square_loss":
            loss_class = square_loss()
        elif self.loss_function == "logistic_loss":
            loss_class = logistic_loss()

        loss_class.construct(output, b, n_input=n_output, n_output=self.n_outputs_)

        self.param_list += loss_class.params

        loss = loss_class.loss_function

        self._predict = theano.function([A], loss_class.y_pred, mode='FAST_RUN')
        self._compute_loss = theano.function([A, b], loss, mode='FAST_RUN')

        # 3. Train Network

        # (1) Get gradient derivatives
        gparams = [T.grad(loss, param) for param in self.param_list]

        # (2) Initialize optimizer
        if self.update_algorithm == "sgd":
            optimizer = sgd_class()
        elif self.update_algorithm == "adagrad":
            pass


        # (3) Create shared datasets
        X_shared = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
        y_shared = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)

        n_train_batches = X_shared.get_value(borrow=True).shape[0] / self.batch_size

        # (4) Initialize optimizer parameters
        optimizer.construct(X_shared, y_shared, A, b, self.param_list, gparams, self.learning_rate,
                            self.batch_size, loss)

        # (5) Optimize the objective function
        t = time.clock()
        for epoch in xrange(self.max_epochs):
            for batch_index in xrange(n_train_batches):
                optimizer.update(epoch=epoch,
                                 batch_index=batch_index)

            if self.verbose:
                print "epoch: %d, loss: %.3f" % (epoch, self._compute_loss(X, y))

        print "time span:", time.clock() - t



        return self


class NeuralNetworkClassifier(BaseNeuralNetwork, ClassifierMixin):
    def __init__(self, layers=[], loss_function="logistic_loss", learning_rate=0.1, batch_size=100, max_epochs=10,
                 update_algorithm="sgd", verbose="False"):

        sup = super(NeuralNetworkClassifier, self)
        sup.__init__(layers=layers, loss_function=loss_function, learning_rate=learning_rate,
                     batch_size=batch_size, max_epcohs=max_epochs, update_algorithm=update_algorithm,
                     verbose=verbose)

        self.label_binarizer_ = LabelBinarizer()

    def fit(self, X, y):
        y_binarized = self.label_binarizer_.fit_transform(y)
        super(NeuralNetworkClassifier, self).fit(X, y_binarized)

    def predict(self, X):
        y_pred =  self._predict(X)

        return self.label_binarizer_.inverse_transform(y_pred)

    def predict_proba(self, X):
        return  self._predict(X)


class NeuralNetworkRegressor(BaseNeuralNetwork, RegressorMixin):
    def __init__(self, layers=[], loss_function="square_loss", learning_rate=1e-5, batch_size=100, max_epochs=10,
                 update_algorithm="sgd", verbose="False"):

        sup = super(NeuralNetworkRegressor, self)
        sup.__init__(layers=layers, loss_function=loss_function, learning_rate=learning_rate,
                     batch_size=batch_size, max_epcohs=max_epochs, update_algorithm=update_algorithm,
                     verbose=verbose)

    def predict(self, X):
        return  self._predict(X)


