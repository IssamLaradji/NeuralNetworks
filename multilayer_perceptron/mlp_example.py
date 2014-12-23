"""
==============================================
Using multilayer perceptron for classification
==============================================

This uses multi-layer perceptron to train on a digits dataset. The example
then reports the training score.
"""

from sklearn.datasets import load_digits

from multilayer_perceptron  import MultilayerPerceptronClassifier

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Create MLP Object
# Please see line 562 in "multilayer_perceptron.py" for more information 
# about the parameters
mlp = MultilayerPerceptronClassifier(hidden_layer_sizes = (50, 20), \
                                     max_iter = 200, alpha = 0.02)

# Train MLP
mlp.fit(X, y)

# Report scores
print "Training Score = ", mlp.score(X,y)
print "Predicted labels = ", mlp.predict(X)
print "True labels = ", y 


