from random_neural_network import RandomNNClassifier
from random_neural_network import RandomNNRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_digits
from itertools import product
from sklearn.utils.testing import assert_greater, assert_array_equal

digits_dataset_multi = load_digits(n_class=3)

Xdigits_multi = MinMaxScaler().fit_transform(digits_dataset_multi.data[:200])
ydigits_multi = digits_dataset_multi.target[:200]

digits_dataset_binary = load_digits(n_class=2)

Xdigits_binary = MinMaxScaler().fit_transform(digits_dataset_binary.data[:200])
ydigits_binary = digits_dataset_binary.target[:200]
classification_datasets = {'binary': (Xdigits_binary, ydigits_binary),
                           'multi-class': (Xdigits_multi, ydigits_multi)}


batch_size = 50
random_state = 1

for dataset, class_weight in product(classification_datasets.values(),
                                     ["balanced", 'auto']):
    X, y = dataset
    randomnn_standard = RandomNNClassifier(class_weight=class_weight,
                                 random_state=random_state, activation='elu')
    randomnn_recursive = RandomNNClassifier(class_weight=class_weight,
                                  random_state=random_state,
                                  batch_size=batch_size, activation='elu')
    randomnn_standard.fit(X, y)
    randomnn_recursive.fit(X, y)

    pred1 = randomnn_standard.predict(X)
    pred2 = randomnn_recursive.predict(X)

    assert_array_equal(pred1, pred2)
    print randomnn_standard.score(X, y)
    assert_greater(randomnn_standard.score(X, y), 0.95)

    from scipy.linalg import sqrtm, inv
    import numpy as np

def nonlin_elu(X, hid_units):
    w_matrix = 2*np.random.random((X.shape[1],hid_units))-1

    #orthonormalize weight matrix
    onorm_w_matrix = w_matrix.dot(inv(sqrtm(w_matrix.T.dot(w_matrix))))
    blob = np.dot(X,onorm_w_matrix)

    #after blob passes through activation finction
    a=1
    blob_activation = np.where(blob>0,blob,a*(np.exp(blob)-1))
    return np.real(blob_activation)

X, y = classification_datasets.values()[0]
print nonlin_elu(X, 50)