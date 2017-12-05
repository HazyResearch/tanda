from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from dataset import read_data_sets
from experiments.train_scripts import flags, select_fold, train
from experiments.tfs.image import *
from functools import partial
from itertools import chain


FLAGS = flags.FLAGS


#####################################################################
# Transformation functions

tfs = list(chain.from_iterable([
    [partial(TF_rotate, angle=p) for p in [2.5, -2.5, 5, -5, 10, -10]],
    [partial(TF_zoom, scale=p) for p in [0.9, 1.1]],
    [partial(TF_shear, shear=p) for p in [0.1, -0.1, 0.2, -0.2, 0.4, -0.4]],
    [partial(TF_swirl, strength=p) for p in [0.1, -0.1, 0.2, -0.2, 0.4, -0.4]],
    [partial(TF_elastic_deform, alpha=p) for p in [1.0, 1.25, 1.5]],
    [TF_erosion, TF_dilation]
]))

#####################################################################

if __name__ == '__main__':

    # Load MNIST data
    dims     = [28, 28, 1]
    DATA_DIR = 'experiments/mnist/data'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    data_iterator = read_data_sets(DATA_DIR, one_hot=True)
    X_train, Y_train = data_iterator.train.images, data_iterator.train.labels
    X_v, Y_v = data_iterator.validation.images, data_iterator.validation.labels
    X_test, Y_test = data_iterator.test.images, data_iterator.test.labels

    if FLAGS.n_folds > 0:
        X_train, Y_train = select_fold(X_train, Y_train)

    # Run training scripts
    train(X_train, dims, tfs, Y_train=Y_train, X_valid=X_v, Y_valid=Y_v,
        n_classes=10)
