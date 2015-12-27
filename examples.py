import numpy as np
from sklearn.datasets import make_classification

from neural_network_theano.layer_classes import fully_connected_layer
from neural_network_theano.neural_network import NeuralNetworkClassifier, NeuralNetworkRegressor

np.random.seed(0)


# Classification
X, y= make_classification(1000, 50, n_informative=10, n_classes=3)

layers = [fully_connected_layer(n_hidden=10), fully_connected_layer(n_hidden=10),
          fully_connected_layer(n_hidden=10)]

nn = NeuralNetworkClassifier(layers=layers, batch_size=1000, max_epochs=1000, learning_rate=1e-4, verbose=True)
nn.fit(X, y)

print "Classification Score %.8f" % nn.score(X, y)


# Regression
X = np.random.rand(1000,50)
y = np.random.rand(1000, 1)

layers = [fully_connected_layer(n_hidden=10), fully_connected_layer(n_hidden=10),
          fully_connected_layer(n_hidden=10)]

nn = NeuralNetworkRegressor(layers=layers, batch_size=1000, max_epochs=1000, learning_rate=1e-5, verbose=True)
nn.fit(X, y)

print "Regression Score %.8f" % nn.score(X, y)
