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
