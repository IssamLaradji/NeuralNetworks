Neural Network Toolbox
======================

This repository contains neural networks implemented in Theano. Theano is a great optimization library that can compile
functions and their gradients. It can also harness the GPU processing power if Theano is configured correctly.

The repository `neural_network_theano` can be used in a similar way as scikit-learn. But since neural networks
can have different layers and update rules, initializing them is just a bit more complicated.

`examples.py` shows how to initialize a classification and a regression model and train them on different datasets.


References
==========

* https://github.com/Theano/Theano
* https://github.com/HIPS/autograd
* https://github.com/scikit-learn/scikit-learn
