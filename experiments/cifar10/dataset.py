from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os

from functools import partial
from six.moves import cPickle
from skimage import img_as_float


def load_cifar10_batch(fpath, one_hot=True, as_float=True):
    with open(fpath, 'rb') as f:
        data = cPickle.load(f, encoding='latin1')
        X = np.copy(data['data']).reshape(-1, 32*32, 3, order='F')
        X = X.reshape(-1, 32, 32, 3)
        Y = np.array(data['labels'])

        # Convert labels to one hot
        if one_hot:
            Y = to_one_hot(Y)

        # CONVERT TO FLOAT [0,1] TYPE HERE to be consistent with skimage TFs!!!
        # See: http://scikit-image.org/docs/dev/user_guide/data_types.html
        if as_float:
            X = img_as_float(X)
    return X, Y


def to_one_hot(y, n_classes=10):
    Y = np.zeros([y.shape[0], n_classes])
    for i in range(y.shape[0]):
        Y[i, y[i]] = 1
    return Y


def load_cifar10_data(data_root, one_hot=True, as_float=True,
    validation_set=True):
    """Load training (first 4 batches), validation (5th batch), and test set.
    If validation_set=False, combines training and validation sets, and returns
    test set as both validation and test.
    """
    # Apply loading format uniformly
    load_batch = partial(load_cifar10_batch, one_hot=one_hot, as_float=as_float)

    # Load training data
    X_train, Y_train = [], []
    train_batches = 4 if validation_set else 5
    for i in range(train_batches):
        X, Y = load_batch(os.path.join(data_root, 'data_batch_%s' % (i+1,)))
        X_train.append(X)
        Y_train.append(Y)
    X_train = np.vstack(X_train)
    Y_train = np.concatenate(Y_train)

    # Load test data
    X_test, Y_test = load_batch(os.path.join(data_root, 'test_batch'))

    # Load validation data
    if validation_set:
        X_valid, Y_valid = load_batch(os.path.join(data_root, 'data_batch_5'))
    else:
        X_valid, Y_valid = X_test, Y_test
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
