from sklearn.datasets import load_digits
from multilayer_perceptron  import MultilayerPerceptronClassifier, MultilayerPerceptronRegressor
import numpy as np
from matplotlib import pyplot as plt

# contrive the "exclusive or" problem
X = np.array([[0,0], [1,0], [0,1], [1,1]])
y = np.array([0, 1, 1, 0])

# MLP training performance
mlp = MultilayerPerceptronClassifier(n_hidden = 5,max_iter = 200, alpha = 0.02)
mlp.fit(X, y)

print "Training Score = ", mlp.score(X,y)
print "Predicted labels = ", mlp.predict(X)
print "True labels = ", y 
# plot decision function

xx, yy = np.meshgrid(np.linspace(-1, 2, 500),
                     np.linspace(-1, 2, 500))
Z = mlp.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, 
          extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
          origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                      linetypes='--')
plt.scatter(X[:, 0], X[:, 1], s=70, c=y, cmap=plt.cm.Paired)

plt.axis([-1, 2, -1, 2])
plt.show()


