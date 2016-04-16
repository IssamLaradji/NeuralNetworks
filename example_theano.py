import numpy as np
from sklearn.datasets import make_classification, make_regression

from neural_network_theano.layer_classes import fully_connected_layer, convolutional_layer
from neural_network_theano.neural_network import NeuralNetworkClassifier, NeuralNetworkRegressor

np.random.seed(0)


# Classification
X, y = make_classification(1000, 64, n_informative=10, n_classes=5)

layers = [convolutional_layer()]

nn = NeuralNetworkClassifier(layers=layers, batch_size=1000, max_epochs=300, learning_rate=1e-4, verbose=True)
nn.fit(X, y)



print "Classification Score %.8f" % nn.score(X, y)


# Regression
X, y = make_regression(1000, 50)

layers = [fully_connected_layer(n_hidden=30)]

nn = NeuralNetworkRegressor(layers=layers, batch_size=10, max_epochs=20, learning_rate=1e-4, verbose=True)
nn.fit(X, y)

print "Regression Score %.8f" % nn.score(X, y)
