from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from .discriminator import Discriminator
from functools import partial


D_H = 2
D_W = 2


class DCNN(Discriminator):
    """
    Discriminator from DCGAN paper
    From https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    """
    def __init__(self, dims=[28, 28, 1], df_dim=64):
        super(DCNN, self).__init__(dims=(dims if len(dims) == 3 else dims+[1]))
        self.df_dim  = df_dim
        self.out_dim = self.last_layer_size

    def _get_logits_op(self, X, n_classes=1, train=True, reuse=False,
        get_layers=False, **kwargs):
        """Returns logits"""
        batch_norm = partial(batch_norm_op, 
            bn_vars_collection=self.bn_vars_collection)
        n_batch = tf.shape(X)[0]
        # Apply convolutional layers
        h0    = conv2d(X, self.dims[-1], self.df_dim, name='d_h0_conv')
        h0_a  = lrelu(h0)
        h1    = conv2d(h0_a, self.df_dim, self.df_dim * 2, name='d_h1_conv')    
        h1_a  = lrelu(batch_norm(h1, name='bn_1', train=train, reuse=reuse))
        h2    = conv2d(h1_a, self.df_dim * 2, self.df_dim * 4, name='d_h2_conv')    
        h2_a  = lrelu(batch_norm(h2, name='bn_2', train=train, reuse=reuse))
        h3    = conv2d(h2_a, self.df_dim * 4, self.df_dim * 8, name='d_h3_conv')    
        h3_a  = lrelu(batch_norm(h3, name='bn_3', train=train, reuse=reuse))
        h_out = tf.reshape(h3_a, [n_batch, self.out_dim])
        h4    = linear(h_out, self.out_dim, n_classes, scope='d_h3_lin')
        # Check for get_layers
        if get_layers:
            layers = [tf.reshape(z, (n_batch, -1)) for z in [h0, h1, h2, h3]]
            return h4, layers
        return h4

    @property
    def last_layer_size(self):
        n_convs, h, w = 4, D_H, D_W
        z1, z2  = self.dims[0], self.dims[1]
        for _ in xrange(n_convs):
            z1, z2 = int(np.ceil(float(z1) / h)), int(np.ceil(float(z2) / w))
        return int(z1 * z2 * self.df_dim * (2. ** (n_convs - 1)))


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def conv2d(X, in_dim, out_dim, k_h=5, k_w=5, d_h=D_H, d_w=D_W, stddev=0.02,
    name="conv2d"):
    # Note: dims is (h, w, n_channels)
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, in_dim, out_dim],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        # Standard convolution
        conv = tf.nn.conv2d(X, w, strides=[1, d_h, d_w, 1], padding='SAME')
        # Add biases
        biases = tf.get_variable('biases', [out_dim],
            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
    return conv


def batch_norm_op(x, bn_vars_collection="BN_vars", train=True, reuse=False,
    epsilon=1e-5, momentum=0.9, name="batch_norm"):
    return tf.contrib.layers.batch_norm(
        x, 
        decay=momentum,
        scale=True,
        epsilon=epsilon, 
        variables_collections=[bn_vars_collection],
        is_training=train,
        reuse=reuse,
        scope=name
    )


def linear(X, in_dim, out_size, scope=None, stddev=0.02, bias_start=0.0):
    with tf.variable_scope(scope or "linear"):
        w = tf.get_variable("w", [in_dim, out_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("bias", [out_size],
                initializer=tf.constant_initializer(bias_start))
    return tf.matmul(X, w) + b
