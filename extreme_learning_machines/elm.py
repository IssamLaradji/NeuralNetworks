"""Extreme Learning Machines
"""

# Author: Issam H. Laradji <issam.laradji@gmail.com>
# Licence: BSD 3 clause

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import pinv2
from sklearn.utils import check_random_state, atleast2d_or_csr
from sklearn.utils import as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import logistic_sigmoid, safe_sparse_dot
import scipy


def _softmax(Z):
    exp_Z = np.exp(Z)
    return (exp_Z.T / exp_Z.sum(axis=1)).T


class BaseELM(BaseEstimator):
    __metaclass__ = ABCMeta

    def __init__(self, n_hidden, regularized):
        #self.regressor = regressor
        self.n_hidden = n_hidden
        self.regularized = regularized

    def _init_fit(self, X):
        """Initialize weight and bias parameters."""
        rng = check_random_state(0)

        self.coef_hidden_ = np.random.normal(
            -1, 1, (self.n_features, self.n_hidden))
        self.intercept_hidden_ = np.random.normal(-1, 1, (1, self.n_hidden))

        #self.coef_output_ = rng.uniform(-1, 1, (self.n_hidden, self.n_outputs))
        #self.intercept_output_ = rng.uniform(-1, 1, self.n_outputs)

    def _get_hidden_activations(self, X):

        A = safe_sparse_dot(X, self.coef_hidden_)

        A += self.intercept_hidden_

        Z = logistic_sigmoid(A)

        return Z

    def _solve_regularized(self, y):
        first = safe_sparse_dot(
            self.hidden_activations_.T, self.hidden_activations_)
        self.coef_output_ = safe_sparse_dot(
            pinv2(first + 1 * np.identity(first.shape[0])), safe_sparse_dot(
                self.hidden_activations_.T, y))

    def _solve(self, y):
        self.coef_output_ = safe_sparse_dot(pinv2(self.hidden_activations_), y)

    def fit(self, X, y):

        n_samples, self.n_features = X.shape
        self.n_outputs = y.shape[1]
        self._init_fit(X)

        self.hidden_activations_ = self._get_hidden_activations(X)

        if self.regularized:
            self._solve_regularized(as_float_array(y, copy=True))
        else:
            self._solve(as_float_array(y, copy=True))

        return self

    def decision_function(self, X):

        X = atleast2d_or_csr(X)

        # compute hidden layer activations
        self.hidden_activations_ = self._get_hidden_activations(X)

        output = safe_sparse_dot(self.hidden_activations_, self.coef_output_)

        return output


class ELMRegressor(BaseELM, RegressorMixin):

    def __init__(self, n_hidden=20, regularized=False):

        super(ELMRegressor, self).__init__(n_hidden, regularized)

        self.hidden_activations_ = None

    def fit(self, X, y):

        super(ELMRegressor, self).fit(X, y)

        return self

    def predict(self, X):

        return self.decision_function(X)


class ELMClassifier(BaseELM, ClassifierMixin):

    def __init__(self, n_hidden=20, regularized=False):

        super(ELMClassifier, self).__init__(n_hidden, regularized)

        self.classes_ = None

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        self._lbin = LabelBinarizer()
        y = self._lbin.fit_transform(y)

        super(ELMClassifier, self).fit(X, y)
        return self

    def predict(self, X):
        X = atleast2d_or_csr(X)
        scores = self.decision_function(X)

        # if len(scores.shape) == 1:
        #scores = logistic_sigmoid(scores)
        #results = (scores > 0.5).astype(np.int)

        # else:
            #scores = _softmax(scores)
            #results = scores.argmax(axis=1)
            # self.classes_[results]
        return self._lbin.inverse_transform(scores)

    def predict_proba(self, X):
        scores = self.decision_function(X)

        if len(scores.shape) == 1:
            scores = logistic_sigmoid(scores)
            return np.vstack([1 - scores, scores]).T
        else:
            return _softmax(scores)
