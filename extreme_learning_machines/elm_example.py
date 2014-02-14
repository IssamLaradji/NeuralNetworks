from sklearn.datasets import load_digits
from elm import ELMClassifier

# testing elm on the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# normalization is important
X/=255

clf = ELMClassifier(n_hidden=30)
clf.fit(X,y)

print 'score:', clf.score(X,y)