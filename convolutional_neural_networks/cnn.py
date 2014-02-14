"""Convolutional Neural Network (still in draft)
   performance does not reflect Convolutional Neural Network capability
"""

# Author: Issam H. Laradji <issam.laradji@gmail.com>
# Licence: BSD 3 clause

import numpy as np

from abc import ABCMeta, abstractmethod
from scipy.optimize import fmin_l_bfgs_b
from scipy.signal import convolve2d, convolve

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.externals import six
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import gen_even_slices
from sklearn.utils import shuffle
from sklearn.utils import atleast2d_or_csr, check_random_state, column_or_1d
from sklearn.utils.extmath import logistic_sigmoid, safe_sparse_dot


def _identity(X):
    """returns the same input array."""
    return X


def _d_logistic(sigm_X):
    """Implements the derivative of the logistic function.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
    """
    return sigm_X * (1 - sigm_X)


def _softmax(Z):
    """Implements the K-way softmax, (exp(Z).T / exp(Z).sum(axis=1)).T

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
    """
    exp_Z = np.exp(Z.T - Z.max(axis=1)).T
    return (exp_Z.T / exp_Z.sum(axis=1)).T


def _tanh(X):
    """Implements the hyperbolic tan function

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
    """
    return np.tanh(X, X)


def _d_tanh(Z):
    """Implements the derivative of the hyperbolic tan function

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
    """
    Z *= Z
    Z *= -1
    Z += 1
    return Z


def _squared_loss(Y, Z):
    """Implements the square loss for regression."""
    return np.sum((Y - Z) ** 2) / (2 * len(Y))


def _log_loss(Y, Z):
    """Implements Logistic loss for binary class.

    Max/Min clipping is enabled to prevent
    invalid zero value in log computation.
    """
    Z = np.clip(Z, 0.00000001, 0.99999999)
    return -np.sum(Y * np.log(Z) +
                  (1 - Y) * np.log(1 - Z)) / Z.shape[0]




def cnnPool(dim_pool, convolved_features):

    n_images = convolved_features.shape[0];
    n_filters = convolved_features.shape[1];
    dim_conv = convolved_features.shape[2];


    pooled_features = np.zeros((n_images, n_filters, dim_conv / dim_pool, dim_conv / dim_pool));

    for id_image in range(n_images):
        for id_feature in range(n_filters):
             for row_pool in range(dim_conv / dim_pool):
                offsetRow = (row_pool) * dim_pool;
                for col_pool in range(dim_conv / dim_pool):
                    offsetCol = (col_pool) * dim_pool;
                    patch = convolved_features[id_image, id_feature, offsetRow: offsetRow + dim_pool, \
                        offsetCol: offsetCol + dim_pool]

                    pooled_features[id_image, id_feature, row_pool, col_pool] = np.mean(patch);

    return pooled_features

class BaseMultilayerPerceptron(six.with_metaclass(ABCMeta, BaseEstimator)):

    """Base class for MLP classification and regression.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
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
        'log_loss': _log_loss,
    }

    @abstractmethod
    def __init__(self, n_filters, dim_filter, dim_pool,
            max_iter, random_state, verbose):

        self.max_iter = max_iter
        self.n_filters = n_filters
        self.dim_filter = dim_filter
        self.dim_pool = dim_pool
        self.random_state = random_state
        self.verbose = verbose

        self.coef_hidden_ = None


    def forward_pass(self, X):

        n_samples = X.shape[0]

        dim_conv = self.dim_image - self.dim_filter + 1; # dimension of convolved output
        dim_output = (dim_conv) / self.dim_pool; # dimension of subsampled output


        a_convolved = np.zeros((n_samples,  self.n_filters, dim_conv, dim_conv));
        a_convolved =  self.cnnConvolve(X, self.coef_conv_pool_, self.intercept_conv_pool_);

        activations_pooled = np.zeros((n_samples, self.n_filters, dim_output,dim_output));

        a_pooled = cnnPool ( self.dim_pool, a_convolved);

        # Reshape activations into 2-d matrix, hiddenSize x numImages,
        # for Softmax layer

        a_pooled = np.reshape(a_pooled,(n_samples, -1));



        a_output = np.zeros((n_samples,  self.n_classes));

       
        a_output = safe_sparse_dot(a_pooled, self.coef_output_) + self.intercept_output_
        a_output = self.output_func(a_output)

        return a_output, a_convolved, a_pooled

    def cnnConvolve(self, X, W, b):
        n_images = X.shape[0];
        dim_images = X.shape[1];
        dim_conv = dim_images - self.dim_filter + 1;

        convolved_features = np.zeros((n_images, self.n_filters, dim_conv, dim_conv));

        for image_num in range(n_images):
          for filter_num in range(self.n_filters):

            convolved_image = np.zeros((dim_conv, dim_conv));

            filter_ = W[filter_num]
            b_ = b[filter_num]

            filter_ = np.rot90(np.squeeze(filter_),2);

            im = np.squeeze(X[image_num]);


            convolved_image = convolve2d(im, filter_, 'valid')

            convolved_image = logistic_sigmoid(convolved_image + b_)
            
            convolved_features[image_num, filter_num] = convolved_image;

        return convolved_features

    def _pack(self, W1, W2, b1, b2):
        """Pack the coefficients and intercepts from theta."""
        return np.hstack((W1.ravel(), W2.ravel(),
                          b1.ravel(), b2.ravel()))

    def _unpack(self, theta):

        outDim = (self.dim_image - self.dim_filter + 1)/self.dim_pool;
        hiddenSize = np.square(outDim)*self.n_filters;
        

        # Reshape theta
        indS = 0;
        indE = np.square(self.dim_filter)*self.n_filters;
        
        self.coef_conv_pool_ = np.reshape(theta[indS:indE], (self.n_filters, self.dim_filter, self.dim_filter));
        indS = indE;
        indE = indE+hiddenSize*self.n_classes;
        self.coef_output_ = np.reshape(theta[indS:indE], (hiddenSize,self.n_classes));
        indS = indE;
        indE = indE+self.n_filters;
        self.intercept_conv_pool_ = theta[indS:indE];
        self.intercept_output_ = theta[indE:];

    def _validate_params(self):
        """Validate input params. """

        if self.max_iter <= 0:
            raise ValueError("max_iter must be > zero")

        # raises ValueError if not registered
        if self.activation not in self.activation_functions:
            raise ValueError("The activation %s"
                             " is not supported. " % self.activation)
        if self.algorithm not in ["sgd", "l-bfgs"]:
            raise ValueError("The algorithm %s"
                             " is not supported. " % self.algorithm)



    def _init_fit(self):

        """Initialize weight and bias parameters."""
        rng = check_random_state(self.random_state)

        np.testing.assert_(self.dim_filter < self.dim_image,'dim_filter must be less that dim_image');

        self.coef_conv_pool_ = 1e-1*np.random.randn(self.dim_filter, self.dim_filter, self.n_filters);

        outDim = self.dim_image - self.dim_filter + 1; # dimension of convolved image

        np.testing.assert_(outDim % self.dim_pool==0,'dim_pool must divide dim_image - dim_filter + 1');

        outDim = outDim/self.dim_pool;
        hiddenSize = np.square(outDim)*self.n_filters;

        # we'll choose weights uniformly from the interval [-r, r]
        r  = np.sqrt(6) / np.sqrt(self.n_classes+hiddenSize+1);
        self.coef_output_ = np.random.rand(self.n_classes, hiddenSize) * 2 * r - r;

        self.intercept_conv_pool_ = np.zeros((self.n_filters, 1));
        self.intercept_output_ = np.zeros((self.n_classes, 1));


    def _init_param(self):
        """Sets the activation, derivative, loss and output functions."""
        self.activation_func = self.activation_functions[self.activation]
        self.derivative_func = self.derivative_functions[self.activation]

        # output for regression
        if self.classes_ is None:
            self.output_func = _identity
        # output for multi class
        elif len(self.classes_) > 2 and self.multi_label is False:
            self.output_func = _softmax
        # output for binary class and multi-label
        else:
            self.output_func = logistic_sigmoid



    def fit(self, X, y):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        y : numpy array of shape (n_samples)
             Subset of the target values.

        Returns
        -------
        self
        """
        self.dim_image = np.sqrt(X.shape[1]);
        self.n_classes = y.shape[1];  


        self._validate_params()
        self._init_fit()
        self._init_param()
        
        X = atleast2d_or_csr(X)
        X = np.reshape(X,(-1, self.dim_image, self.dim_image)); 

        
        self.n_outputs = y.shape[1]

        self._backprop_lbfgs(X, y)

        return self


    def _cnn_cost(self, theta, X, y):


        dim_image = X.shape[1] # height/width of image
        n_samples = X.shape[0]; # number of images
        #alpha = 3e-3 

        self._unpack(theta);

        # Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
        coef_conv_pool_grad = np.zeros(self.coef_conv_pool_.shape);
        coef_output_grad = np.zeros(self.coef_output_.shape);
        intercept_conv_pool_grad = np.zeros(self.intercept_conv_pool_.shape);
        intercept_output_grad = np.zeros(self.intercept_output_.shape);


        a_output, a_convolved, a_pooled = self.forward_pass(X)

        dim_conv = self.dim_image - self.dim_filter + 1; # dimension of convolved output
        dim_output = (dim_conv) / self.dim_pool; # dimension of subsampled output

        cost = _log_loss(y, a_output)
        #print cost
        #cost = (-1/n_samples)*np.sum((np.log(a_output)* y))

        # add regularization term to cost
        #cost += (0.5 * alpha) * (np.sum(coef_conv_pool_ ** 2) +
        #                                  np.sum(coef_output_ ** 2)) \
        #        / n_samples


        # Compute the error at the output layer (softmax)
        delta_output_ = a_output - y

        #delta_d = - (y- probs); 
        #delta_s = Wd_p * delta_d; #% the Pooling / Sample Layer ' s preactivation
        #delta_s = reshape (delta_s, outputDim, outputDim, numFilters, numImages); Wc, Wd, bc, bd
        
        # Back propogate errors from softmax layer to pooling layer
        delta_pool_ = np.dot(delta_output_, self.coef_output_.T)
        delta_pool_ = np.reshape (delta_pool_, (n_samples,  self.n_filters, dim_output, dim_output));
        #erewr
        #print coef_conv_pool_.shape, delta_pooled_.shape
        delta_convolved_ =  (1./np.square( self.dim_pool)) * np.kron(delta_pool_ , \
            np.ones(( self.dim_pool, self.dim_pool)))* _d_logistic(a_convolved)

        coef_output_grad = ( 1. / n_samples) * safe_sparse_dot(a_pooled.T, delta_output_)\
                           #+ alpha * coef_output_; 

        intercept_output_grad = ( 1. / n_samples) * np.sum(delta_output_, 0);#% Note that this is required and


        for i in range( self.n_filters):
            WC_I = np.zeros(( self.dim_filter,  self.dim_filter))
            for j in range(n_samples):
                WC_I = WC_I + convolve2d(X[j], np.rot90((delta_convolved_[j,i]), 2 ), 'valid' );
         
            #WC_I = convolve(X, np.rot90(delta_convolved_[:,i],2),'valid' )

            coef_conv_pool_grad[i] = ( 1. / n_samples) * WC_I# + alpha * coef_conv_pool_[i];
            
            intercept_pool_ = delta_convolved_[:,i];

            intercept_conv_pool_grad[i] = np.sum(intercept_pool_) / n_samples;

        # Unroll gradient into grad vector for minFunc
        grad = np.hstack((coef_conv_pool_grad.ravel(), coef_output_grad.ravel(),
                              intercept_conv_pool_grad.ravel(), intercept_output_grad.ravel()));

        return cost, grad

   

    def _backprop_lbfgs(self, X, y):

        initial_theta = self._pack(
            self.coef_conv_pool_,
            self.coef_output_,
            self.intercept_conv_pool_,
            self.intercept_output_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1


        optTheta, _, _ = fmin_l_bfgs_b(
            func= self._cnn_cost,
            x0=initial_theta,
            maxfun=self.max_iter,
            iprint=iprint,
            args=(
                X,
                y))

        self._unpack(optTheta)


    def decision_function(self, X):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples)
        Predicted target values per element in X.
        """
        X = atleast2d_or_csr(X)

        X = np.reshape(X,(-1, self.dim_image, self.dim_image)); 

        output,_,_ = self.forward_pass(X)

        if output.shape[1] == 1:
            output = output.ravel()

        return output


class ConvolutionalNeuralNetworkClassifier(BaseMultilayerPerceptron,
                                     ClassifierMixin):


    def __init__(
            self, n_filters=2, 
            dim_filter = 9, dim_pool = 2,
            max_iter=200, random_state=None,
            verbose=False):

        sup = super(ConvolutionalNeuralNetworkClassifier, self)
        sup.__init__(n_filters, dim_filter, dim_pool,
            max_iter, random_state, verbose)

        self.loss = 'log_loss'
        self.algorithm = 'l-bfgs'
        self.activation = 'logistic'
        self.classes_ = None

    def fit(self, X, y):

        y = column_or_1d(y, warn=True)

        # needs a better way to check multi-label instances
        if isinstance(np.reshape(y, (-1, 1))[0][0], list):
            self.multi_label = True
        else:
            self.multi_label = False

        self.classes_ = np.unique(y)
        self._lbin = LabelBinarizer()
        y = self._lbin.fit_transform(y)

        super(ConvolutionalNeuralNetworkClassifier, self).fit(X, y)

        return self

  

    def predict(self, X):
        """Predict using the multi-layer perceptron model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples)
            Predicted target values per element in X.
        """
        X = atleast2d_or_csr(X)
        scores = self.decision_function(X)

        if len(scores.shape) == 1 or self.multi_label is True:
            scores = logistic_sigmoid(scores)
            results = (scores > 0.5).astype(np.int)

            if self.multi_label:
                return self._lbin.inverse_transform(results)

        else:
            scores = _softmax(scores)
            results = scores.argmax(axis=1)

        return self.classes_[results]

    def predict_log_proba(self, X):
        """Returns the log of probability estimates.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        T : array-like, shape (n_samples, n_outputs)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in
            `self.classes_`. Equivalent to log(predict_proba(X))
        """
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        """Probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples, n_outputs)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        scores = self.decision_function(X)

        if len(scores.shape) == 1:
            scores = logistic_sigmoid(scores)
            return np.vstack([1 - scores, scores]).T
        else:
            return _softmax(scores)


class ConvolutionalNeuralNetworkRegressor(BaseMultilayerPerceptron, RegressorMixin):

    
    def __init__(
            self, n_hidden=100, activation="logistic",
            algorithm='l-bfgs', alpha=0.00001,
            batch_size=200, learning_rate="constant", eta0=0.1,
            power_t=0.25, max_iter=100, shuffle=False,
            random_state=None, tol=1e-5,
            verbose=False, warm_start=False):

        sup = super(MultilayerPerceptronRegressor, self)
        sup.__init__(n_hidden, activation,
                     algorithm, alpha,
                     batch_size, learning_rate,
                     eta0, power_t,
                     max_iter, shuffle,
                     random_state,
                     tol, verbose,
                     warm_start)

        self.loss = 'squared_loss'
        self.classes_ = None

    def fit(self, X, y):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        y : numpy array of shape (n_samples)
            Subset of the target values.

        Returns
        -------
        self
        """
        y = np.atleast_1d(y)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        super(MultilayerPerceptronRegressor, self).fit(X, y)
        return self

    def partial_fit(self, X, y):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        y : numpy array of shape (n_samples)
            Subset of the target values.

        Returns
        -------
        self
        """
        y = np.atleast_1d(y)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        super(MultilayerPerceptronRegressor, self).partial_fit(X, y)
        return self

    def predict(self, X):
        """Predict using the multi-layer perceptron model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples)
            Predicted target values per element in X.
        """
        X = atleast2d_or_csr(X)

        return self.decision_function(X)
