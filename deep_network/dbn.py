"""Deep Belief Network
"""

# Author: Issam H. Laradji <issam.laradji@gmail.com>
# Licence: BSD 3 clause

import numpy as np
from scipy.linalg import norm

from abc import ABCMeta, abstractmethod
from sklearn.utils import gen_even_slices
from sklearn.utils import shuffle
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import atleast2d_or_csr, check_random_state
from sklearn.utils.extmath import logsumexp, safe_sparse_dot, logistic_sigmoid
from itertools import cycle, izip
from sklearn.neural_network import Autoencoder, BernoulliRBM
from sklearn.linear_model import LogisticRegression


def _d_logistic(X):
    """Implements the derivative of the logistic function.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape = [n_samples, n_features]
    """
    return X * (1 - X)


def _log_softmax(X):
    """Implements the logistic K-way softmax, (exp(X).T / exp(X).sum(axis=1)).T,
    in the log domain

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape = [n_samples, n_features]
    """
    return (X.T - logsumexp(X, axis=1)).T


def _softmax(X):
    """Implements the K-way softmax, (exp(X).T / exp(X).sum(axis=1)).T

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape = [n_samples, n_features]
    """
    exp_X = np.exp(X)
    return (exp_X.T / exp_X.sum(axis=1)).T


def _tanh(X):
    """Implements the hyperbolic tan function

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape = [n_samples, n_features]
    """
    return np.tanh(X, X)


def _d_tanh(X):
    """Implements the derivative of the hyperbolic tan function

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape = [n_samples, n_features]
    """
    X *= -X
    X += 1
    return X


def _logistic_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred.clip(min=0.00000001)) + (1 - y_true) * np.log(1 - y_pred.clip(max=0.99999999)))


def _squared_loss(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / (2 * len(y_true))


def _log_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))


class BaseDBN(BaseEstimator):

    """Base class for MLP classification and regression.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    __metaclass__ = ABCMeta

    activation_functions = {
        'tanh': _tanh,
        'logistic': logistic_sigmoid,
        'softmax': _softmax
    }
    derivative_functions = {
        'tanh': _d_tanh,
        'logistic': _d_logistic
    }
    loss_functions = {
        'squared_loss': _squared_loss,
        'log': _log_loss,
        'logistic_log': _logistic_loss
    }

    @abstractmethod
    def __init__(
        self, n_hidden, activation, loss, algorithm,
            alpha, batch_size, learning_rate, eta0, power_t,
            max_iter, shuffle_data, random_state, tol, warm_start, verbose):
        self.activation = activation
        self.loss = loss
        self.algorithm = algorithm
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.max_iter = max_iter
        self.n_hidden = n_hidden
        self.shuffle_data = shuffle_data
        self.random_state = random_state
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose

    def _pack(self, W1, W2, b1, b2):
        """Pack the coefficients and intercepts from theta"""
        return np.hstack((W1.ravel(), W2.ravel(),
                          b1.ravel(), b2.ravel()))

    def _unpack(self, theta, n_features, n_outputs):
        """Extracts the coefficients and intercepts from theta

        Parameters
        ----------
        theta : array-like, shape (size(W1)*size(W2)*size(b1)*size(b2), 1)
            Contains concatenated flattened weights  that represent the parameters "W1, W2, b1, b2"
        n_features : int
            Number of features
        n_outputs : int
            Number of output neurons
        """
        N = self.n_hidden * n_features
        N2 = n_outputs * self.n_hidden
        self.coef_hidden_ = np.reshape(theta[:N], (n_features, self.n_hidden))
        self.coef_output_ = np.reshape(
            theta[N:N2 + N], (self.n_hidden, n_outputs))
        self.intercept_hidden_ = theta[N2:N2 + self.n_hidden]
        self.intercept_output_ = theta[N2 + N + self.n_hidden:]

    def _init_fit(self, X, y, n_features, n_outputs):
        """Initialize weight and bias parameters

        Parameters
        ----------
        n_features : int
            Number of features

        n_outputs : int
            Number of output neurons

        """
        n_hidden = self.n_hidden
        rng = check_random_state(self.random_state)
        # init
        self.coef_hidden_ = [0] * self.n_layers
        self.intercept_hidden_ = [0] * self.n_layers
        #
        Autoencoder
        ae = Autoencoder(
            n_hidden=100,
            decoder='linear',
            random_state=3)
        # print 'Getting stage', 1, 'features...'

        ae = Autoencoder(n_hidden=n_hidden[0], decoder='linear').fit(X)
        ae_features = ae.transform(X)
        self.coef_hidden_[0] = ae.coef_hidden_
        self.intercept_hidden_[0] = ae.intercept_hidden_
        # traverse the layers
        for layer in xrange(1, self.n_layers):
            # print 'Getting stage', layer+1, 'features...'
            ae = Autoencoder(
                n_hidden=n_hidden[layer], decoder='linear').fit(ae_features)
            ae_features = ae.transform(ae_features)
            self.coef_hidden_[layer] = ae.coef_hidden_
            self.intercept_hidden_[layer] = ae.intercept_hidden_
        lr = LogisticRegression()
        # print 'Getting stage', self.n_layers+1, 'features...'
        lr.fit(ae_features, self._lbin.inverse_transform(y))
        self.coef_output_ = lr.coef_.T
        self.intercept_output_ = lr.intercept_
        """
        self.coef_hidden_[0] = rng.uniform(-1, 1, (n_features, n_hidden[0]))
        self.coef_output_ = rng.uniform(-1, 1, (n_hidden[-1], n_outputs))
        self.intercept_hidden_[0] = rng.uniform(-1, 1, n_hidden[0])
        self.intercept_output_ = rng.uniform(-1, 1, n_outputs)
        #traverse the layers
        for layer in xrange(1, self.n_layers):
            self.coef_hidden_[layer] = rng.uniform(-1, 1, (n_hidden[layer-1], n_hidden[layer]))
            self.intercept_hidden_[layer] = rng.uniform(-1, 1, n_hidden[layer])
        """

    def _init_param(self):
        """Sets the activation, derivative and the output functions"""
        self.activation_func = self.activation_functions[self.activation]
        self.derivative_func = self.derivative_functions[self.activation]
        # output for regression
        if self.classes_ is None:
            self.output_func = lambda X: X
        # output for multi class
        elif len(self.classes_) > 2:
            self.output_func = _softmax
        # output for binary class
        else:
            self.output_func = logistic_sigmoid

    def fit(self, X, Y):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Y : numpy array of shape [n_samples]
            Subset of the target values.

        Returns
        -------
        self
        """
        self.n_layers = len(self.n_hidden)
        X = atleast2d_or_csr(X, dtype=np.float64, order="C")
        n_outputs = Y.shape[1]
        n_samples, n_features = X.shape
        self._init_fit(X, Y, n_features, n_outputs)
        self._init_param()
        if self.shuffle_data:
            X, Y = shuffle(X, Y, random_state=self.random_state)
        self.batch_size = np.clip(self.batch_size, 0, n_samples)
        n_batches = n_samples / self.batch_size
        batch_slices = list(
            gen_even_slices(
                n_batches *
                self.batch_size,
                n_batches))
        # l-bfgs does not work well with batches
        if self.algorithm == 'l-bfgs':
            self.batch_size = n_samples
        # preallocate memory
        a_hidden = [0] * self.n_layers
        a_output = np.empty((self.batch_size, n_outputs))
        delta_o = np.empty((self.batch_size, n_outputs))
        # print 'Fine tuning...'
        if self.algorithm is 'sgd':
            eta = self.eta0
            t = 1
            prev_cost = np.inf
            for i in xrange(self.max_iter):
                for batch_slice in batch_slices:
                    cost, eta = self.backprop_sgd(
                        X[batch_slice],
                        Y[batch_slice],
                        self.batch_size,
                        a_hidden,
                        a_output,
                        delta_o,
                        t,
                        eta)
                if self.verbose:
                        print("Iteration %d, cost = %.2f"
                              % (i, cost))
                if abs(cost - prev_cost) < self.tol:
                    break
                prev_cost = cost
                t += 1
        elif 'l-bfgs':
                self._backprop_lbfgs(
                    X, Y, n_features, n_outputs, n_samples, a_hidden,
                    a_output,
                    delta_o)
        return self

    def backprop(self, X, Y, n_samples, a_hidden, a_output, delta_o):
        """Computes the MLP cost  function ``J(W,b)``
        and the corresponding derivatives of J(W,b) with respect to the
        different parameters given in the initialization

        Parameters
        ----------
        theta: array-like, shape (size(W1)*size(W2)*size(b1)*size(b2))
            Contains concatenated flattened weights  that represent the parameters "W1, W2, b1, b2"
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        n_features : int
            Number of features
        n_outputs : int
            Number of output neurons
        n_samples : int
            Number of samples

        Returns
        -------
        cost: float
        grad: array-like, shape = [(size(W1)*size(W2)*size(b1)*size(b2)), ]

        """
        # Forward propagate
        """
        a_hidden[:] = self.activation_func(safe_sparse_dot(X, self.coef_hidden_) +
                                      self.intercept_hidden_)
        """
        a_hidden[0] = self.activation_func(safe_sparse_dot(X, self.coef_hidden_[0]) +
                                           self.intercept_hidden_[0])
        # traverse the layers
        for layer in xrange(1, self.n_layers):
            a_hidden[layer] = self.activation_func(safe_sparse_dot(a_hidden[layer - 1], self.coef_hidden_[layer]) +
                                                   self.intercept_hidden_[layer])
        a_output[:] = self.output_func(safe_sparse_dot(a_hidden[-1], self.coef_output_) +
                                       self.intercept_output_)
        # Backward propagate
        diff = Y - a_output
        delta_o[:] = -diff
        delta_h = [0] * self.n_layers
        delta_h[-1] = np.dot(delta_o, self.coef_output_.T) *\
            self.derivative_func(a_hidden[-1])
        # traverse the layers
        for layer in xrange(self.n_layers - 2, -1, -1):
            delta_h[layer] = np.dot(delta_h[layer + 1], self.coef_hidden_[layer + 1].T) *\
                self.derivative_func(a_hidden[layer])
        cost = self.loss_functions[self.loss](Y, a_output)
        # Add regularization term to cost
        cost += (
            0.5 * self.alpha) * (
            np.sum(
                np.sum(np.array([np.sum(c ** 2) for c in self.coef_hidden_]))) + np.sum(
                self.coef_output_ ** 2))
        # init
        grad_hidden_ = [0] * self.n_layers
        intercept_grad_hidden_ = [0] * self.n_layers
        #
        # Get regularized gradient
        grad_hidden_[0] = (safe_sparse_dot(X.T, delta_h[0]) +
                           (self.alpha * self.coef_hidden_[0])) * (1.0 / n_samples)
        grad_output_ = (safe_sparse_dot(a_hidden[-1].T, delta_o) +
                        (self.alpha * self.coef_output_)) * (1.0 / n_samples)
        intercept_grad_hidden_[0] = np.mean(delta_h[0], 0)
        # traverse the layers
        for layer in xrange(1, self.n_layers):
            grad_hidden_[layer] = (safe_sparse_dot(a_hidden[layer - 1].T, delta_h[layer]) +
                                   (self.alpha * self.coef_hidden_[layer])) * (1.0 / n_samples)
            intercept_grad_hidden_[layer] = np.mean(delta_h[layer], 0)
        intercept_grad_output_ = np.mean(delta_o, 0)
        return cost, grad_hidden_, grad_output_, intercept_grad_hidden_, intercept_grad_output_

    def backprop_sgd(
            self, X, Y, n_samples, a_hidden, a_output, delta_o, t, eta):
        """Updates the weights using the computed gradients

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Y : numpy array of shape [n_samples]
            Subset of the target values.

        n_features: int
            Number of features

        n_outputs: int
            Number of output neurons

        n_samples: int
            Number of samples

        """
        cost,  grad_hidden_, grad_output_, intercept_grad_hidden_, intercept_grad_output_ = self.backprop(
            X, Y, n_samples, a_hidden, a_output, delta_o)
        # Update weights
        # traverse the layers
        for layer in xrange(self.n_layers):
            self.coef_hidden_[layer] -= (eta * grad_hidden_[layer])
        self.coef_output_ -= (eta * grad_output_)
        # traverse the layers
        for layer in xrange(self.n_layers):
            self.intercept_hidden_[
                layer] -= (eta * intercept_grad_hidden_[layer])
        self.intercept_output_ -= (eta * intercept_grad_output_)
        if self.learning_rate == 'optimal':
            eta = 1.0 / (self.alpha * t)
        elif self.learning_rate == 'invscaling':
            eta = self.eta0 / pow(t, self.power_t)
        return cost, eta

    def _backprop_lbfgs(
            self, X, Y, n_features, n_outputs, n_samples,
            a_hidden, a_output, delta_o):
        """Applies the quasi-Newton optimization methods that uses a l_BFGS
        to train the weights

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Y : numpy array of shape [n_samples]
            Subset of the target values.

        n_features : int
            Number of features

        n_outputs : int
            Number of output neurons

        n_samples : int
            Number of samples

        """
        initial_theta = self._pack(
            self.coef_hidden_,
            self.coef_output_,
            self.intercept_hidden_,
            self.intercept_output_)
        optTheta, _, _ = fmin_l_bfgs_b(
            func=self._cost_grad,
            x0=initial_theta,
            maxfun=self.max_iter,
            disp=self.verbose,
            args=(
                X,
                Y,
                n_features,
                n_outputs,
                n_samples,
                a_hidden, a_output, delta_o))
        self._unpack(optTheta, n_features, n_outputs)

    def _cost_grad(self, theta, X, Y, n_features, n_outputs,
                   n_samples, a_hidden, a_output, delta_o):
        """Computes the MLP cost  function ``J(W,b)``
        and the corresponding derivatives of J(W,b) with respect to the
        different parameters given in the initialization

        Parameters
        ----------
        theta : array-like, shape (size(W1)*size(W2)*size(b1)*size(b2))
            Contains concatenated flattened weights  that represent the parameters "W1, W2, b1, b2"
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        n_features : int
            Number of features
        n_outputs : int
            Number of output neurons
        n_samples : int
            Number of samples

        Returns
        -------
        cost : float
        grad : array-like, shape (size(W1)*size(W2)*size(b1)*size(b2))

        """
        self._unpack(theta, n_features, n_outputs)
        cost, W1grad, W2grad, b1grad, b2grad = self.backprop(
            X, Y, n_samples, a_hidden, a_output, delta_o)
        grad = self._pack(W1grad, W2grad, b1grad, b2grad)
        return cost, grad

    def partial_fit(self, X, y, classes):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Subset of training data

        y : numpy array of shape [n_samples]
            Subset of target values

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : returns an instance of self.
        """
        X = atleast2d_or_csr(X, dtype=np.float64, order="C")
        _, n_features = X.shape
        self._init_param()
        if self.classes_ is None and classes is None:
            raise ValueError("classes must be passed on the first call "
                             "to partial_fit.")
        elif classes is not None and self.classes_ is not None:
            if not np.all(self.classes_ == np.unique(classes)):
                raise ValueError("`classes` is not the same as on last call "
                                 "to partial_fit.")
        elif classes is not None:
            self._lbin = LabelBinarizer(classes=classes)
            Y = self._lbin.fit_transform(y)
            self._init_fit(n_features, Y.shape[1])
        else:
            Y = self._lbin.transform(y)
        self.backprop_naive(X, Y, 1)
        return self

    def decision_function(self, X):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples]
           Predicted target values per element in X.
        """
        a_hidden = self.activation_func(safe_sparse_dot(X, self.coef_hidden_[0]) +
                                        self.intercept_hidden_[0])
        # traverse the layers
        for layer in xrange(1, self.n_layers):
            a_hidden = self.activation_func(safe_sparse_dot(a_hidden, self.coef_hidden_[layer]) +
                                            self.intercept_hidden_[layer])
        output = safe_sparse_dot(a_hidden, self.coef_output_) +\
            self.intercept_output_
        if output.shape[1] == 1:
            output = output.ravel()
        return output


class DBNClassifier(BaseDBN, ClassifierMixin):

    """Multi-layer perceptron (feedforward neural network) classifier.

    Trained with gradient descent under the loss function which is estimated
    for each sample batch at a time and the model is updated along the way
    with a decreasing strength schedule (aka learning rate).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector.

    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values for the features.

    Parameters
    ----------
    n_hidden : int
        Number of units in the hidden layer.
        
    activation: string, optional
        Activation function for the hidden layer; either "logistic" for
        1 / (1 + exp(x)), or "tanh" for the hyperbolic tangent.
        
    loss: 'logistic_log', or 'log'
        The loss function to be used. Defaults to 'squared_loss' for Regression
        and 'log' for Classification
        
    alpha : float, optional
        L2 penalty (regularization term) parameter.
        
    batch_size : int, optional
        Size of minibatches in SGD optimizer.
        
    learning_rate : float, optional
        Base learning rate for weight updates. 
        
    max_iter : int, optional
        Maximum number of iterations.
        
    random_state : int or RandomState, optional
        State of or seed for random number generator.
        
    shuffle : bool, optional
        Whether to shuffle samples in each iteration before extracting
        minibatches.
        
    tol : float, optional
        Tolerance for the optimization. When the loss at iteration i+1 differs
        less than this amount from that at iteration i, convergence is
        considered to be reached.
        
    eta0 : double, optional
        The initial learning rate [default 0.01].
        
    power_t : double, optional
        The exponent for inverse scaling learning rate [default 0.25].
        
    verbose : bool, optional
        Whether to print progress messages to stdout.

    """

    def __init__(
        self, n_hidden=[100], activation="logistic",
        loss='log', algorithm='l-bfgs', alpha=0.00001, batch_size=1000,
        learning_rate="constant", eta0=0.8, power_t=0.5, max_iter=200,
            shuffle_data=False, random_state=None, tol=1e-5, warm_start=False,  verbose=False):
        super(
            DBNClassifier, self).__init__(n_hidden, activation, loss,
                                          algorithm, alpha, batch_size, learning_rate, eta0,
                                          power_t, max_iter, shuffle_data, random_state, tol, warm_start, verbose)
        self.classes_ = None

    def fit(self, X, y):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Y : numpy array of shape [n_samples]
            Subset of the target values.

        Returns
        -------
        self
        """
        self.classes_ = np.unique(y)
        self._lbin = LabelBinarizer()
        Y = self._lbin.fit_transform(y)
        if len(self.classes_) == 2:
            self.loss = 'logistic_log'
        super(DBNClassifier, self).fit(
            X, Y)
        return self

    def predict(self, X):
        """Predict using the multi-layer perceptron model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples]
           Predicted target values per element in X.
        """
        X = atleast2d_or_csr(X)
        scores = super(DBNClassifier, self).decision_function(X)
        if len(scores.shape) == 1:
            scores = logistic_sigmoid(scores)
            indices = (scores > 0.5).astype(np.int)
        else:
            scores = _softmax(scores)
            indices = scores.argmax(axis=1)
        return self._lbin.classes_[indices]

    def predict_log_proba(self, X):
        """Log of probability estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_outputs]
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in
            `self.classes_`.
        """
        scores = super(DBNClassifier, self).decision_function(X)
        if len(scores.shape) == 1:
            return np.log(self.activation_func(scores))
        else:
            return _log_softmax(scores)

    def predict_proba(self, X):
        """Probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples, n_outputs]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        scores = super(DBNClassifier, self).decision_function(X)
        if len(scores.shape) == 1:
            scores = logistic_sigmoid(scores)
            return np.vstack([1 - scores, scores]).T
        else:
            return _softmax(scores)


class DBNRegressor(BaseDBN, RegressorMixin):

    """Multi-layer perceptron (feedforward neural network) classifier.

    Trained with gradient descent under the loss function which is estimated
    for each sample batch at a time and the model is updated along the way
    with a decreasing strength schedule (aka learning rate).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector.

    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values for the features.

    Parameters
    ----------
    n_hidden : int
        Number of units in the hidden layer.
        
    activation: string, optional
        Activation function for the hidden layer; either "logistic" for
        1 / (1 + exp(x)), or "tanh" for the hyperbolic tangent.
        
    loss: 'logistic_log', or 'log'
        The loss function to be used. Defaults to 'squared_loss' for Regression
        and 'log' for Classification
        
    alpha : float, optional
        L2 penalty (regularization term) parameter.
        
    batch_size : int, optional
        Size of minibatches in SGD optimizer.
        
    learning_rate : float, optional
        Base learning rate for weight updates. 
        
    max_iter : int, optional
        Maximum number of iterations.
        
    random_state : int or RandomState, optional
        State of or seed for random number generator.
        
    shuffle : bool, optional
        Whether to shuffle samples in each iteration before extracting
        minibatches.
        
    tol : float, optional
        Tolerance for the optimization. When the loss at iteration i+1 differs
        less than this amount from that at iteration i, convergence is
        considered to be reached.
        
    eta0 : double, optional
        The initial learning rate [default 0.01].
        
    power_t : double, optional
        The exponent for inverse scaling learning rate [default 0.25].
        
    verbose : bool, optional
        Whether to print progress messages to stdout.

    """

    def __init__(
        self, n_hidden=[100], activation="logistic",
        loss='squared_loss', algorithm='l-bfgs', alpha=0.00001, batch_size=200,
        learning_rate="constant", eta0=0.8, power_t=0.5, max_iter=200,
            shuffle_data=False, random_state=None, tol=1e-5, warm_start=False, verbose=False):
        super(
            DBNRegressor, self).__init__(n_hidden, activation, loss,
                                         algorithm, alpha, batch_size, learning_rate, eta0,
                                         power_t, max_iter, shuffle_data, random_state, tol, warm_start, verbose)
        self.classes_ = None

    def fit(self, X, y):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Y : numpy array of shape [n_samples]
            Subset of the target values.

        Returns
        -------
        self
        """
        if len(y.shape) == 1:
            y = y[:, np.newaxis]
        super(DBNRegressor, self).fit(
            X, y)
        return self

    def predict(self, X):
        """Predict using the multi-layer perceptron model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples]
           Predicted target values per element in X.
        """
        X = atleast2d_or_csr(X)
        return super(DBNRegressor, self).decision_function(X)
