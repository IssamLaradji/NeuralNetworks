"""
====================================================
Using Sparse Autoencoder features for classification
====================================================

This compares the performance of stochastic gradient descent (SGD) on raw 
image pixels and on Autoencoder extracted features .
"""

import random
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from multilayer_perceptron  import MultilayerPerceptronAutoencoder
from sklearn.utils import shuffle


random.seed(100)

# Download the MNIST dataset and grab 200 images for testing
mnist = fetch_mldata('MNIST original')
X, y = mnist.data, mnist.target
X, y = shuffle(X, y, random_state=0)
indices = np.array(random.sample(range(70000), 2000))
X, y = X[indices].astype('float64'), y[indices]
X = X / 255

ae = MultilayerPerceptronAutoencoder(
    algorithm='l-bfgs',
    verbose=True,
    max_iter=200,
    hidden_layer_sizes=100,
    random_state=3,
    batch_size=X.shape[0])

# Train autoencoder and extract features
ae_features = ae.fit_transform(X)

clf = SGDClassifier(random_state=3)

# Train Classifier on raw pixel features and report score
clf.fit(X, y)
print 'SGD on raw pixels score: ', clf.score(X, y)

# Train Classifier on Autoencoder features and report score
clf.fit(ae_features, y)
print 'SGD on extracted features score: ', clf.score(ae_features, y)

