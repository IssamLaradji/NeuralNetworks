import numpy as np
from sklearn.datasets import load_digits

from neural_network_theano.layer_classes import fully_connected_layer, convolutional_layer
from neural_network_theano.neural_network import NeuralNetworkClassifier, NeuralNetworkRegressor

np.random.seed(0)


# Classification
data = load_digits()

X, y = data.data, data.target
X /= 255.

layers = [convolutional_layer(), fully_connected_layer()]

nn = NeuralNetworkClassifier(layers=layers, batch_size=200, max_epochs=200, learning_rate=5e-6, verbose=True)
nn.fit(X, y)

print "Classification Score %.8f" % nn.score(X, y)
