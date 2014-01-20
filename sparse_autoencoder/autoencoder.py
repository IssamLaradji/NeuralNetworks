"""Sparse Autoencoder
"""

# Author: Issam Laradji <issam.laradji@gmail.com>
# Licence: BSD 3 clause

import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import norm
from itertools import cycle, izip
from sklearn.utils import atleast2d_or_csr, check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils import shuffle
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import logistic_sigmoid, safe_sparse_dot


def _identity(X):
    """returns the same input array."""
    return X
    
def _binary_KL_divergence(p, p_hat):
    """
    Computes the a real, KL divergence of two binomial distributions with
    probabilities p  and p_hat respectively.
    """
    return (p * np.log(p / p_hat)) + ((1 - p) * np.log((1 - p) / (1 - p_hat)))

def _d_logistic(X):
    """
    Implements the derivative of the logistic function.

    Parameters
    ----------
    x: array-like, shape (M, N)

    Returns
    -------
    x_new: array-like, shape (M, N)
    """
    return X * (1 - X)


class Autoencoder(BaseEstimator, TransformerMixin):

    """
    Sparse Autoencoder (SAE)

    A Sparse Autoencoder with one hidden layer.
    Parameters
    ----------
    n_hidden : int, default 100
        Number of units in the hidden layer.

    algorithm : {'l-bfgs', 'sgd'}, default 'l-bfgs'
        The algorithm for weight optimization.  Defaults to 'l-bfgs'

        - 'l-bfgs' is an optimization algorithm in the family of quasi-
           Newton methods.

        - 'sgd' refers to stochastic gradient descent.
        
    learning_rate : {'constant', 'invscaling'}, default 'constant'
        Base learning rate for weight updates.

        -'constant', as it stands,  keeps the learning rate 'eta' constant
          throughout training. eta = eta0

        -'invscaling' gradually decreases the learning rate 'eta' at each
          time step 't' using an inverse scaling exponent of'power_t'.
          eta = eta0 / pow(t, power_t)
    
    eta0 : double, optional, default 0.5
        The initial learning rate used. It controls the step-size
        in updating the weights.

    power_t : double, optional, default 0.25
        The exponent for inverse scaling learning rate.
        It is used in updating eta0 when the learning_rate
        is set to 'invscaling'.
        
    beta : float, default 3
        Weight of sparsity penalty term
        
    sparsity_param : float, default 0.1
        Desired average activation of the hidden units
        
    batch_size : int, default 500
        Number of examples per minibatch.
        
    max_iter : int, default 200
        Number of iterations/sweeps over the training dataset to perform
        during training.
        
    tol : float, default 1e-5
        Tolerance for the optimization. When the loss at iteration i+1 differs
        less than this amount from that at iteration i, convergence is
        considered to be reached.
        
    verbose : bool, default False
        When True (False by default) the method outputs the progress
        of learning after each iteration.
        
    random_state : integer or numpy.RandomState, default None
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    self.coef_hidden_ : array-like, shape (n_hidden, n_features)
        Weight matrix, where n_features in the number of visible
        units and n_hidden is the number of hidden units.
        
    self.coef_output_  : array-like, shape (n_features, n_hidden)
        Weight matrix, where n_features in the number of visible
        units and n_hidden is the number of hidden units.
        
    intercept_hidden_  : array-like, shape (n_hidden,), optional
        Biases of the hidden units
        
    intercept_visible_  : array-like, shape (n_features,), optional
        Biases of the visible units

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.neural_network import SAE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = SAE(n_hidden=10)
    >>> model.fit(X)
    Autoencoder(activation_func='logistic', alpha=0.0001, batch_size=1000, beta=3,
  learning_rate=0.0001, max_iter=20, n_hidden=10,
  algorithm='l-bfgs', random_state=None, sparsity_param=0.01,
  tol=1e-05, verbose=False)

    References
    ----------

    [1] Ngiam, Jiquan, et al. "On optimization methods for deep learning."
        Proceedings of the 28th International Conference on Machine Learning (ICML-11). 2011.
        http://ai.stanford.edu/~quocle/LeNgiCoaLahProNg11.pdf
    """
    def __init__(
        self, n_hidden=25, algorithm='l-bfgs',
        decoder = 'non_linear', alpha=3e-3, beta=3, sparsity_param=0.1,
        batch_size=500, shuffle_data=False, max_iter=200, tol=1e-5, learning_rate="constant", eta0=0.5, 
        power_t = 0.25, verbose=False, random_state=None):
            
        self.algorithm = algorithm
        self.decoder = decoder
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.beta = beta
        self.sparsity_param = sparsity_param
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        
        self.activation_func = logistic_sigmoid
        self.derivative_func = _d_logistic
        
    def _init_fit(self, n_features):
        """Initialize weight and bias parameters."""
        rng = check_random_state(self.random_state)
        
        self.coef_hidden_ = rng.uniform(-1, 1, (n_features, self.n_hidden))
        self.coef_output_ = rng.uniform(-1, 1, (self.n_hidden, n_features))
        self.intercept_hidden_ = rng.uniform(-1, 1, self.n_hidden)
        self.intercept_output_ = rng.uniform(-1, 1, n_features)

    def _init_param(self):
        """Sets the activation, derivative, loss and output functions."""      
        # output for non-linear
        if self.decoder=='non_linear':
            self.output_func =  logistic_sigmoid
            self.output_derivative = _d_logistic
        # output for linear
        if self.decoder=='linear':
            self.output_func = _identity
            self.output_derivative = _identity
            
    def _init_t_eta_(self):
        """Initialize iteration counter attr ``t_``"""
        self.t_ = 1.0
        self.eta_ = self.eta0
        
    def _unpack(self, theta, n_features):
        """
        Extract the coefficients and intercepts (W1,W2,b1,b2) from theta

        Parameters
        ----------
        theta : array-like, shape (size(W1)*size(W2)*size(b1)*size(b2), 1)
          Contains concatenated flattened weights  that represent the parameters "W1, W2, b1, b2"
        n_features : int
          Number of features (visible nodes).
        """
        N = self.n_hidden * n_features
        self.coef_hidden_ = np.reshape(theta[:N],
                                      (n_features, self.n_hidden))
        self.coef_output_ = np.reshape(theta[N:2 * N],
                                      (self.n_hidden, n_features))
        self.intercept_hidden_ = theta[2 * N:2 * N + self.n_hidden]
        self.intercept_output_ = theta[2 * N + self.n_hidden:]

    def _pack(self, W1, W2, b1, b2):
        """
        Pack the coefficients and intercepts (W1,W2,b1,b2) from theta

        Parameters
        ----------
        theta : array-like, shape (size(W1)*size(W2)*size(b1)*size(b2), 1)
            Contains concatenated flattened weights  that represent the parameters "W1, W2, b1, b2"
            
        n_features : int
            Number of features
        """
        return np.hstack((W1.ravel(), W2.ravel(),
                          b1.ravel(), b2.ravel()))

    def transform(self, X):
        """
        Computes the extracted features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
        """
        return self.activation_func(safe_sparse_dot(X, self.coef_hidden_) + self.intercept_hidden_)

    def fit_transform(self, X, y=None):
        """
        Fit the model to the data X and transform it.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        """
        self.fit(X)
        return self.transform(X)

    def fit(self, X, y=None):
        """
        Fit the model to the data X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self
        """
        X = atleast2d_or_csr(X, dtype=np.float64, order="C")
        n_samples, n_features = X.shape
        self._init_fit(n_features)
        self._init_param()
        self._init_t_eta_()
        
        if self.shuffle_data:
            X, y = shuffle(X, y, random_state=self.random_state)
            
        # l-bfgs does not use mini-batches
        if self.algorithm == 'l-bfgs':
            batch_size = n_samples
        else:
            batch_size = np.clip(self.batch_size, 0, n_samples)
            n_batches = n_samples / batch_size
            batch_slices = list(
                gen_even_slices(
                    n_batches * batch_size,
                    n_batches))
            
        # preallocate memory
        a_hidden = np.empty((batch_size, self.n_hidden))
        a_output = np.empty((batch_size, n_features))
        delta_o = np.empty((batch_size, n_features))
        
        if self.algorithm == 'sgd':
            prev_cost = np.inf
            
            for i in xrange(self.max_iter):
                    for batch_slice in batch_slices:
                        cost = self.backprop_sgd(
                            X[batch_slice],
                            n_features, self.batch_size,
                            delta_o, a_hidden, a_output)
                    if self.verbose:
                        print("Iteration %d, cost = %.2f"
                              % (i, cost))
                    if abs(cost - prev_cost) < self.tol:
                        break
                    prev_cost = cost
                    self.t_ += 1
                              
        elif self.algorithm == 'l-bfgs':
            self._backprop_lbfgs(
                X, n_features,
                a_hidden, a_output, 
                delta_o, n_samples)
                
        return self

    def backprop(self, X, n_features, n_samples,
                  delta_o, a_hidden, a_output):
        """
        Computes the sparse autoencoder cost  function
        and the corresponding derivatives of  with respect to the
        different parameters given in the initialization [1]

        Parameters
        ----------
        theta : array-like, shape (size(W1)*size(W2)*size(b1)*size(b2))
          Contains concatenated flattened weights  that represent the parameters "W1, W2, b1, b2"
          
        X : array-like, shape (n_samples, n_features)
          Training data, where n_samples in the number of samples
          and n_features is the number of features.
          
        n_features : int
          Number of features (visible nodes).
          
        n_samples : int
          Number of samples

       Returns
       -------
       cost : float
       grad : array-like, shape (size(W1)*size(W2)*size(b1)*size(b2))

       References
       -------
       [1] http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity
        """
        # Forward propagate
        a_hidden[:] = self.activation_func(safe_sparse_dot(X, self.coef_hidden_)
                                      + self.intercept_hidden_)
        
        a_output[:] = self.output_func(safe_sparse_dot(a_hidden, self.coef_output_)
                                      + self.intercept_output_)

        # Get average activation of hidden neurons
        sparsity_param_hat = np.sum(a_hidden, 0) / n_samples
        sparsity_delta  = self.beta * \
            ((1 - self.sparsity_param) / (1 - sparsity_param_hat)
             - self.sparsity_param / sparsity_param_hat)
             
        # Backward propagate
        diff = X - a_output
        #Linear decoder
        delta_o[:] = -diff * self.output_derivative(a_output)

        delta_h = (
            (safe_sparse_dot(delta_o, self.coef_output_.T) +
             sparsity_delta)) *\
            self.derivative_func(a_hidden)
            
        # Get cost 
        cost = np.sum(diff ** 2) / (2 * n_samples)
        
        # Add regularization term to cost 
        cost += (0.5 * self.alpha) * (
            np.sum(self.coef_hidden_ ** 2) + np.sum(
                self.coef_output_ ** 2))
                
        # Add sparsity term to the cost
        cost += self.beta * np.sum(
            _binary_KL_divergence(
                self.sparsity_param,
                sparsity_param_hat))
                
        #Get gradients
        W1grad = safe_sparse_dot(X.T, delta_h) / n_samples 
        W2grad = safe_sparse_dot(a_hidden.T, delta_o) / n_samples
        b1grad = np.sum(delta_h, 0) / n_samples
        b2grad = np.sum(delta_o, 0) / n_samples
        
        # Add regularization term to gradients 
        W1grad += self.alpha * self.coef_hidden_
        W2grad += self.alpha * self.coef_output_
        
        return cost, W1grad, W2grad, b1grad, b2grad

    def reconstruct(self, a_hidden):
      
        a_output = self.activation_func(safe_sparse_dot(a_hidden, self.coef_output_)
                                      + self.intercept_output_)
                                      
        return a_output[:]
        
        
    def backprop_sgd(
            self, X, n_features, n_samples, delta_o, a_hidden, a_output):
        """
        Updates the weights using the computed gradients

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Y : numpy array of shape [n_samples]
            Subset of the target values.

        n_features : int
            Number of features

        n_samples : int
            Number of samples

        """
        cost, W1grad, W2grad, b1grad, b2grad = self.backprop(
            X, n_features, n_samples, delta_o, a_hidden, a_output)
            
        # Update weights
        self.coef_hidden_ -= (self.eta_ * W1grad)
        self.coef_output_ -= (self.eta_ * W2grad)
        self.intercept_hidden_ -= (self.eta_ * b1grad)
        self.intercept_output_ -= (self.eta_ * b2grad)

        if self.learning_rate == 'invscaling':
            self.eta_ = self.eta0 / pow(self.t_, self.power_t)
        return cost
        
    def _backprop_lbfgs(
            self, X, n_features, a_hidden, a_output, delta_o, n_samples):
        """
        Applies the one of the optimization methods (l-bfgs-b, bfgs, newton-cg, cg)
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
                n_features,
                n_samples,
                delta_o,
                a_hidden,
                a_output))
        self._unpack(optTheta, n_features)

    def _cost_grad(self, theta, X, n_features,
                   n_samples, delta_o, a_hidden, a_output):
        """
        Computes the  cost  function
        and the corresponding derivatives with respect to the
        different parameters given in the initialization

        Parameters
        ----------
        theta: array-like, shape (size(W1)*size(W2)*size(b1)*size(b2))
            Contains concatenated flattened weights  that represent the parameters "W1, W2, b1, b2"
            
        X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
            
        n_features : int
            Number of features

        n_samples : int
            Number of samples

        Returns
        -------
        cost : float
        grad : array-like, shape (size(W1)*size(W2)*size(b1)*size(b2))

        """
        self._unpack(theta, n_features)
        cost, W1grad, W2grad, b1grad, b2grad = self.backprop(
            X, n_features, n_samples, delta_o, a_hidden, a_output)
            
        return cost, self._pack(W1grad, W2grad, b1grad, b2grad)
