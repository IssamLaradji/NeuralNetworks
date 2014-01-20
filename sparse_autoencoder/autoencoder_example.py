from sklearn.datasets import fetch_mldata
import random
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import Autoencoder
from sklearn.neural_network import MultilayerPerceptronClassifier

random.seed(100)

# download dataset
mnist = fetch_mldata('MNIST original')
X, y = mnist.data, mnist.target

indices = np.array(random.sample(range(70000), 3000))
X, y = X[indices].astype('float64'), y[indices]
# for non-linear autoencoder, feature values in the range [0, 1] is necessary

X = X / 255

ae = Autoencoder(
    algorithm='l-bfgs',
    verbose=True,
    max_iter=200,
    n_hidden=150,
    random_state=3)

# extract features
ae_features = ae.fit_transform(X)

clf = SGDClassifier(random_state=3)
clf.fit(X, y)

# extracted features should have higher score
print 'SGD on raw pixels score: ', clf.score(X, y)
clf.fit(ae_features, y)
print 'SGD on extracted features score: ', clf.score(ae_features, y)

