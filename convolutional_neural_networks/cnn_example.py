"""(in draft mode) scores do not reflect Convolutional Neural Network capability
"""
from sklearn.datasets import load_digits
from cnn import ConvolutionalNeuralNetworkClassifier

# testing elm on the digits dataset
digits = load_digits(2)
X, y = digits.data, digits.target

# normalization is important
X/=255

clf = ConvolutionalNeuralNetworkClassifier(verbose = True,\
                        max_iter = 30, dim_filter = 2, dim_pool = 1)
clf.fit(X,y)

print 'score:', clf.score(X,y)