# Authors: Issam H. Laradji <issam.laradji@gmail.com>
# Licence: BSD 3 clause

import theano

from theano.compat import OrderedDict
import theano.tensor as T
import numpy as np


class sgd_class():
    def construct(self, X_shared, y_shared, A, b, params, gparams, learning_rate, batch_size,loss):
        self.learning_rate = learning_rate

        index = T.lscalar()
        lr = T.dscalar()

        learning_rate = T.dscalar('learning_rate')

        # Create update rule
        updates = OrderedDict()
        for param, gparam in zip(params, gparams):
            updates[param] = param - learning_rate * gparam

        # Construct update function
        self.train_model = theano.function(inputs=[index, lr], updates=updates, outputs=loss,
                                           givens={A: X_shared[index * batch_size: (index + 1) * batch_size],
                                                   b: y_shared[index * batch_size: (index + 1) * batch_size],
                                                   learning_rate: lr})


    def update(self, epoch, batch_index):
        #self.learning_rate /= 2.
        return self.train_model(batch_index, self.learning_rate)



