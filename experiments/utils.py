from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import numpy as np
import os
import sys
import tensorflow as tf

from collections import Counter, defaultdict
from scipy.misc import imsave
from six import iteritems
from subprocess import check_output
from time import strftime


def num_procs_open(procs):
    k = 0
    for p in procs:
        k += (p.poll() is None)
    return k


def create_subdirs(log_path, sub_dir, run_index):
     # Create subdirectory paths
    log_path = os.path.join(log_path, sub_dir)
    log_path = os.path.join(log_path, str(run_index))
    LOGDIR   = os.path.join(log_path, 'logs')
    PLOTDIR  = os.path.join(log_path, 'plots')
    SAVEDIR  = os.path.join(log_path, 'checkpoints')

    # Create directories
    for path in [PLOTDIR, LOGDIR, SAVEDIR]:
        if not os.path.exists(path):
            os.makedirs(path)
    return LOGDIR, PLOTDIR, SAVEDIR


def save_run_log(log_dict, logdir, name='run_log.json'):
    # Save to file and return log_dict
    with open(os.path.join(logdir, name), 'w') as f:
        json.dump(log_dict, f, sort_keys=True, indent=4)


def get_git_revision_short_hash():
    return str(check_output(['git', 'rev-parse', '--short', 'HEAD'])).strip()


def create_run_log(logdir, flags, name='run_log.json'):
    """Creates and saves initial run log including all flags"""
    log_dict = defaultdict(list)
    # Make sure flags are parsed, then use as initial log dict
    import sys
    flags(sys.argv)
    # flags._parse_flags()
    log_dict.update(flags.__flags)
    # Get the git commit hash
    log_dict['commit_hash'] = get_git_revision_short_hash()
    save_run_log(log_dict, logdir, name=name)
    return log_dict


class ConstantPlotter(object):
    """Helper for adding summary op for a constant value outside of the graph"""
    def __init__(self, name, dtype=tf.float32):
        self._build_op(name, dtype)

    def _build_op(self, name, dtype):
        self.x = tf.placeholder(dtype, [])
        x_summary = tf.summary.scalar(name, self.x)
        self.merged = tf.summary.merge([x_summary])

    def get_summary(self, session, x):
        return session.run(self.merged, feed_dict={self.x: x})


class ImagePlotter(object):
    """
    Create ops for inserting images into Tensorboard
    """
    def __init__(self, dims, batch_size, plot_dir, plot_names):
        self.dims       = dims if len(dims) == 3 else dims + [1]
        self.batch_size = batch_size
        self.plot_dir   = plot_dir
        self.plot_names = plot_names
        self._build_plot_ops()
    
    def _build_plot_ops(self):
        N = len(self.plot_names)
        # Compute image height--plotting grid of images using utils.save_image
        h_plot = self.dims[0] * int(np.floor(np.sqrt(self.batch_size)))
        plot_dims = [h_plot, h_plot, self.dims[-1]]

        # Create placeholders
        self.plots = [tf.placeholder(tf.float32, plot_dims) for _ in range(N)]

        # Create image summary ops
        plot_summs = [tf.summary.image(name, tf.reshape(plot, [1] + plot_dims))
            for plot, name in zip(self.plots, self.plot_names)]

        # Build merged summary op
        self.merged_plots = tf.summary.merge(plot_summs)

    def get_image_summaries(self, session, Xs, file_name):
        ps = [save_images(X, '%s_%s' % (file_name, name), self.plot_dir,
            dims=self.dims) for X, name in zip(Xs, self.plot_names)]
        return session.run(self.merged_plots, dict(zip(self.plots, ps)))


def save_images(imgs, name, savedir=None, dims=[28, 28, 1]):
    # Get square number of images
    d = int(np.floor(np.sqrt(len(imgs))))
    X = np.array(imgs[0 : d*d])
    X = X.reshape([d, d] + dims)
    
    # Reshape into grid
    X = np.concatenate(np.split(X, d, axis=0), axis=3)
    X = np.concatenate(np.split(X, d, axis=1), axis=2)
    
    # Save and return
    if savedir is not None:
        imsave(os.path.join(savedir, '{0}.png'.format(name)), np.squeeze(X))
    return X.reshape([d * dims[0], d * dims[1], dims[2]])


def get_log_dir_path(root_path, run_name):
    """
    Creates log dir of format e.g.:
        experiments/log/2017_01_01/run_name_12_00_00/
    """
    date_stamp = strftime("%Y_%m_%d")
    time_stamp = strftime("%H_%M_%S")

    # Group logs by day first
    log_path = os.path.join(root_path, date_stamp)

    # Then, group by run_name and hour + min + sec to avoid duplicates
    log_path = os.path.join(log_path, "_".join([run_name, time_stamp]))
    return log_path


def get_class_idxs(Y):
    class_idxs = defaultdict(list)
    for i, c in enumerate(np.argmax(Y, axis=1)):
        class_idxs[c].append(i)
    return class_idxs


def balanced_subsample(X, Y, n_per_class):
    """
    Returns subsample balanced by class, with max n_per_class
    data points per class, given data matrix X and one-hot label
    matrix Y.
    """
    # Sort by class
    class_idxs = get_class_idxs(Y)

    # Select indices
    balanced_idxs = []
    for c, idxs in iteritems(class_idxs):
        np.random.shuffle(idxs)
        balanced_idxs += idxs[:n_per_class]

    # Shuffle and return
    np.random.shuffle(balanced_idxs)
    return X[balanced_idxs, :], Y[balanced_idxs, :]


def line_writer(od, newline=True):
    """
    Takes in an OrderedDict of (key, value) pairs
    and prints tab-separated with carriage return.
    """
    print_tuples = []
    for k, v in iteritems(od):
        if type(v) == str:
            print_tuples.append('{0} {1}'.format(k, v))
        else:
            print_tuples.append('{0}={1:.4f}'.format(k, v))
    #msg = "\r%s\n" if newline else "\r%s"
    msg = "%s\n" if newline else "\r%s"
    sys.stdout.write(msg % '\t'.join(print_tuples))
    sys.stdout.flush()


################
# Config parsing
################


def parse_str(s):
    """Try to parse a string to int, then float, then bool, then leave alone"""
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            if s.lower() == 'true':
                return True
            if s.lower() == 'false':
                return False
            return s


def parse_config_str(config_str):
    """Convert a string with format kw1=val1,kw2=val2,... to a dict"""
    if len(config_str) == 0:
        return dict()
    kv = [kv_pair.split('=') for kv_pair in config_str.split(',')]
    return dict((k, parse_str(v)) for k, v in kv)


def create_config_str(config):
    """Create a string with format kw1=val1,kw2=val2,... from a dict"""
    return ','.join('{0}={1}'.format(k, v) for k, v in iteritems(config))


###
### DIVERSITY ANALYSIS
###

def jaccard_multiset_distance(x, y):
    """Compute Jaccard multiset distance. 0 is most similar, 1 is least."""
    return 1. - float(sum((x & y).values())) / sum((x | y).values())


def average_all_pairs_jaccard_distance(X):
    """Average Jaccard multiset distance between all pairs of rows of X"""
    sets = [Counter(row.tolist()) for row in X]
    dist = [
        jaccard_multiset_distance(s, t)
        for i, s in enumerate(sets) for t in sets[i+1:]
    ]
    return np.mean(dist)


def get_ngrams(X, n=3):
    """Return unique ngrams occuring in rows of X"""
    ngrams = set()
    for row in X:
        ngrams.update(zip(*[row[i:] for i in range(n)]))
    return ngrams

def ngram_ratio(X, n_symbols, n=3):
    """Ratio of number of unique ngrams in X to total possible"""
    n_ng  = len(get_ngrams(X, n))
    total = min(n_symbols ** n, X.shape[0] * (X.shape[1] - n + 1))
    return float(n_ng) / total
