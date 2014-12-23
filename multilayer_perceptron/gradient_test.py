"""
Testing for Multi-layer Perceptron module (sklearn.neural_network)
"""

# Author: Issam H. Laradji
# Licence: BSD 3 clause
import numpy as np

from numpy.testing import assert_almost_equal, assert_array_equal

from sklearn.datasets import load_digits, load_boston
from multilayer_perceptron  import MultilayerPerceptronClassifier
from multilayer_perceptron  import MultilayerPerceptronRegressor
from multilayer_perceptron  import MultilayerPerceptronAutoencoder
from sklearn.preprocessing import LabelBinarizer

def test_gradient():
    """Test gradient.

    This makes sure that the activation functions and their derivatives
    are correct. The approximated and the real gradients
    should be close.

    """
    X = np.array([[0.3, 0.2, 0.1, 0.3], [0.4, 0.6, 0.43, 0.2], [0.1, 0.2, 0.3, 0.6], \
                  [0.4, 2., 3., 4.]])


    for activation in ['logistic', 'relu', 'tanh']:
        # Create MLP List
        mlps = [('regressor', MultilayerPerceptronRegressor(activation=activation,
                                              hidden_layer_sizes=10, max_iter=1)),
                ('classifier', MultilayerPerceptronClassifier(activation=activation,
                                               hidden_layer_sizes=10, max_iter=1)),
                ('autoencoder', MultilayerPerceptronAutoencoder(
                                                hidden_layer_sizes=10, max_iter=1))
               ]

        for name, mlp in mlps:
            if name == 'autoencoder':
                y = X.copy()
                Y = X.copy()
            else:
                y = np.array([1, 1, 0, 0])
                Y = LabelBinarizer().fit_transform(y)

            mlp.fit(X, y)

            theta = np.hstack([l.ravel() for l in mlp.layers_coef_ +
                               mlp.layers_intercept_])

            layer_units = ([X.shape[1]] + [mlp.hidden_layer_sizes] +
                           [mlp.n_outputs_])

            activations = []
            deltas = []
            coef_grads = []
            intercept_grads = []

            activations.append(X)
            for i in range(mlp.n_layers_ - 1):
                activations.append(np.empty((X.shape[0],
                                             layer_units[i + 1])))
                deltas.append(np.empty((X.shape[0],
                                        layer_units[i + 1])))

                fan_in = layer_units[i]
                fan_out = layer_units[i + 1]
                coef_grads.append(np.empty((fan_in, fan_out)))
                intercept_grads.append(np.empty(fan_out))

            # analytically compute the gradients
            cost_grad_fun = lambda t: mlp._cost_grad_lbfgs(t, X, Y, activations,
                                                           deltas, coef_grads,
                                                           intercept_grads)
            [_, real_gradient] = cost_grad_fun(theta)
            approximated_gradient = np.zeros(np.size(theta))
            n = np.size(theta, 0)
            perturb = np.zeros(theta.shape)
            epsilon = 1e-6
            # numerically compute the gradients
            for i in range(n):
                # dtheta = E[:, i] * epsilon
                # print dtheta
                perturb[i] = epsilon
                approximated_gradient[i] = (cost_grad_fun(theta + perturb)[0] -
                              cost_grad_fun(theta - perturb)[0]) / (epsilon * 2.0)
                perturb[i] = 0

            assert_almost_equal(approximated_gradient, real_gradient)
    print "Gradient Test Passed!"

test_gradient()