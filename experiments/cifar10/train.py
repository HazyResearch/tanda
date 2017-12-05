from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from dataset import load_cifar10_data
from experiments.train_scripts import flags, select_fold, train
from experiments.tfs.image import *
from functools import partial
from itertools import chain


#####################################################################
flags.DEFINE_boolean("validation_set", True, 
    "If False, use validation set as part of training set")

FLAGS = flags.FLAGS
#####################################################################


#####################################################################
# Transformation functions

tfs = list(chain.from_iterable([
    [partial(TF_rotate, angle=p) for p in [2.5, -2.5, 5, -5]],
    [partial(TF_zoom, scale=p) for p in [0.9, 1.1, 0.75, 1.25]],
    [partial(TF_shear, shear=p) for p in [0.1, -0.1, 0.25, -0.25]],
    [partial(TF_swirl, strength=p) for p in [0.1, -0.1, 0.25, -0.25]],
    [partial(TF_shift_hue, shift=p) for p in [0.1, -0.1, 0.25, -0.25]],
    [partial(TF_enhance_contrast, p=p) for p in [0.75, 1.25, 0.5, 1.5]],
    [partial(TF_enhance_brightness, p=p) for p in [0.75, 1.25, 0.5, 1.5]],
    [partial(TF_enhance_color, p=p) for p in [0.75, 1.25, 0.5, 1.5]],
    [TF_horizontal_flip]
]))
#####################################################################

if __name__ == '__main__':

    # Load CIFAR10 data
    dims     = [32, 32, 3]
    DATA_DIR = 'experiments/cifar10/data/cifar-10-batches-py'
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_cifar10_data(
        DATA_DIR, validation_set=FLAGS.validation_set)

    if FLAGS.n_folds > 0:
        X_train, Y_train = select_fold(X_train, Y_train)

    # Run training scripts
    train(X_train, dims, tfs, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid,
        n_classes=10)
