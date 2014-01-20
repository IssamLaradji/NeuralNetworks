from sklearn.datasets import fetch_mldata
import random
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import Autoencoder
from sklearn.neural_network import DBNClassifier


mnist = fetch_mldata('MNIST original')
X, y = mnist.data, mnist.target
random.seed(100)
indices = np.array(random.sample(range(70000), 1000))
X, y = X[indices].astype('float64'), y[indices]
# For SAE, feature values in the range [0, 1] is necessary
X = X / 255
clf = SGDClassifier(random_state=3)
clf.fit(X, y)
ae = Autoencoder(
    algorithm='l-bfgs',
    verbose=True,
    max_iter=200,
    n_hidden=150,
    random_state=3)

ae_features = ae.fit_transform(X)
print 'Logistic Regression on raw pixels score: ', clf.score(X, y)
clf.fit(ae_features, y)
print 'Logistic Regression on Sparse Autoencoder features score: ', clf.score(ae_features, y)
clf = DBNClassifier(n_hidden=[50, 50], algorithm='sgd',
                    activation='logistic', random_state=0).fit(X, y)
print 'Deep Belief Network score: ', clf.score(X, y)
