from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf


ADAM = tf.train.AdamOptimizer


class Discriminator(object):
    """
    Parent class for discriminator in TAN module
    Also includes methods to build supervised version so can be reused
    as end discriminative model.
    """
    def __init__(self, dims=None):
        self.dims = dims        
        # Placeholders for supervised version of discriminator
        self.X        = None
        self.Y        = None
        self.loss     = None
        self.train_op = None
        self.accuracy = None
        self.bn_vars_collection = "BN_vars"

    def _get_logits_op(self, X, n_classes, train=True, reuse=False,
        get_layers=False, **kwargs):
        """Implement this method with sub-class; X has shape [-1] + self.dims"""
        raise NotImplementedError()

    def get_logits_op(self, X, n_classes=1, train=True, reuse=False,
        per_img_std=False, get_layers=False, **kwargs):
        """Returns logits using self._get_logits_op, first preprocessing"""
        X = tf.reshape(X, [-1] + self.dims)
        if per_img_std:
            X = tf.map_fn(tf.image.per_image_standardization, X)
        out = self._get_logits_op(X, n_classes, train=train, reuse=reuse,
            get_layers=get_layers, **kwargs)
        if get_layers and len(out) != 2:
            raise Exception("Specified get_layers but not available")
        return out

    @property
    def last_layer_size(self):
        raise NotImplementedError()

    def get_loss_op(self, logits, name=None, positive=True, mean=True):
        """Loss op for use in TAN (n_classes=1)"""
        y    = tf.ones_like(logits) if positive else tf.zeros_like(logits)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
        return tf.reduce_mean(loss, name=name) if mean else tf.squeeze(loss)

    def _tr_term(self, logits_arr, Np):
        """Get the TR reg term given a loits_arr consisting of Np
        different logits (number of classes = K) of transformations of batches
        of size B. This term is just the average squared distance between the
        logits of a pair of passes for a data point, averaged over the batch.
        
        See https://papers.nips.cc/paper/6333-regularization-with-stochastic-
        transformations-and-perturbations-for-deep-semi-supervised-learning.pdf
        """
        # Reshape to [B, Np, K]
        A = tf.transpose(logits_arr.stack(), [1, 0, 2])

        # ||a_{ij}||_2^2; note element-wise multiply here
        R = tf.reshape(tf.reduce_sum(A * A, 2), [-1, Np, 1])
        # ||a_{ji}||_2^2
        R_t = tf.transpose(R, [0, 2, 1])
        # a_{ij}a_{ji}
        S = tf.matmul(A, tf.transpose(A, [0, 2, 1]))
        # Pairwise distance matrix (a_{ij} - a_{ji})^2
        D = R - 2 * S + R_t

        # Lower triangular part (don't double count)
        D_lt = tf.matrix_band_part(D, -1, 0)
        # Take mean across over distinct pairs & batch size
        return tf.reduce_mean(tf.reduce_sum(D_lt, axis=2))

    def build_supervised(self, n_classes, name, trainer=ADAM, lr_init=0.01,
        per_img_std=True, weight_decay=0.0, ls_term=0.0, ls_term_n_passes=1):
        """Build model for supervised setting
        
        @per_img_std: Per image normalization
        @weight_decay: If > 0.0, adds `self.get_weight_decay_op()` to loss
        @ls_term: Local smoothness term which adds the mean L2 norm of a batch
            of unlabeled data and its transformed copy to minimize as well;
            if > 0.0, adds this op to the graph & training step
        """
        size = np.prod(self.dims)
        summaries = []

        # Note we take *flattened* data (just because that's how TAN uses)
        self.X = tf.placeholder(tf.float32, [None, size])
        self.Y = tf.placeholder(tf.float32, [None, n_classes])
        with tf.variable_scope(name):
            logits = self.get_logits_op(self.X, n_classes=n_classes,
                        per_img_std=per_img_std)
        
        # Loss function
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.Y))
        
        # Add ="weight decay" term if applicable
        if weight_decay > 0:
            self.loss += weight_decay * self.get_weight_decay_op()

        ## TRANSFORMATION REGULARIZATION TERM
        self.U_ts = None
        if ls_term > 0.0:
            # Note that U_ts also includes the un-transformed image, so we do
            # an "extra" pass for this one 
            Np = ls_term_n_passes + 1
            self.U_ts = tf.placeholder(tf.float32, [Np, None, size])

            # Pass through the network
            # NOTE: Any random ops e.g. dropout, etc. should be used here in
            # train mode!
            # NOTE: We regularize logits (not prediction vector); this works
            # much better empirically
            logits_u_t_arr = tf.TensorArray(tf.float32, Np)
            with tf.variable_scope(name, reuse=True):
                
                # Add several transformed versions' logits
                for i in range(Np):
                    logits_u_t = self.get_logits_op(
                                    self.U_ts[i, :, :],
                                    n_classes=n_classes,
                                    per_img_std=per_img_std,
                                    train=True,
                                    reuse=True
                                )
                    logits_u_t_arr = logits_u_t_arr.write(i, logits_u_t)
            
            # Add TR reg term to loss
            u_reg_loss = self._tr_term(logits_u_t_arr, Np)
            summaries.append(tf.summary.scalar("U_loss", u_reg_loss))
            self.loss += ls_term * u_reg_loss

        # Learning rate- constant variable that we can overwrite
        self.lr = tf.constant(lr_init, tf.float32)
        summaries.append(tf.summary.scalar("learning_rate", self.lr))

        # Get summaries
        summaries.append(tf.summary.scalar("loss", self.loss))
        self.train_summaries = tf.summary.merge(summaries)

        # Training step
        var_list = [v for v in tf.trainable_variables()
            if v.name.startswith(name)]
        # Note: This is necessary for batch_norm to be handled correctly
        update_ops = [u for u in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if u.name.startswith(name)]
        with tf.control_dependencies(update_ops):
            self.train_op = trainer(self.lr).minimize(self.loss, 
                var_list=var_list)

        # Accuracy
        # Note: We need to get logits again because we need to set train=False
        # for e.g. batch_norm, dropout, etc.
        with tf.variable_scope(name, reuse=True):
            logits_test = self.get_logits_op(self.X, n_classes=n_classes,
                            per_img_std=per_img_std, train=False, reuse=True)
        
        # Precision computation ops + summary
        correct = tf.equal(tf.argmax(logits_test, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        self.acc_summary = tf.summary.merge([acc_summary])

        # For returning marginal probabilities
        self.probs = tf.nn.softmax(logits_test)

        # For saving variables
        vars = tf.trainable_variables()
        vars_list = filter(lambda v : v.name.startswith(name), vars)
        # Note: In order to save batch_norm moving averages--which are needed
        # at test time--need to save them in a collection (see README) with
        # name self.bn_vars_collection, o/w won't get saved!
        vars_list += tf.get_collection_ref(self.bn_vars_collection)
        self.saver = tf.train.Saver(var_list=vars_list)

    def supervised_train_step(self, session, X, Y, U_ts=None, lr=None):
        feed_dict = {self.X: X, self.Y: Y}
        if self.U_ts is not None and U_ts is not None:
            feed_dict.update({self.U_ts: U_ts})
        if lr is not None:
            feed_dict[self.lr] = lr
        loss, summary, _ = session.run([self.loss, self.train_summaries,
            self.train_op], feed_dict=feed_dict)
        return loss, summary

    def get_accuracy(self, session, X, Y, batch_size=100):
        # NOTE: We do eval in minibatches otherwise too much memory!!
        N = X.shape[0]
        n_batches = int(np.floor(N / batch_size))
        # Iterate over batches
        accs_sum = 0.0
        for i, b in enumerate(range(0, N, batch_size)):

            # Get next batch
            X_batch = X[b : b + batch_size, :]
            Y_batch = Y[b : b + batch_size, :]

            # Get accuracy
            n_batch = X_batch.shape[0]
            batch_acc = session.run(self.accuracy,
                feed_dict={self.X: X_batch, self.Y: Y_batch})
            accs_sum += n_batch * batch_acc

        # Acc = (n_1 * acc_1 + ... + n_k * acc_k) / sum(n_i)
        acc = accs_sum / float(N)
        value = tf.summary.Summary.Value(tag="accuracy", simple_value=acc)
        summary = tf.summary.Summary(value=[value])
        return acc, summary

    def get_probs(self, session, X):
        return session.run(self.probs, {self.X: X})

    def save(self, sess, path):
        """Note this saves _only_ the end model."""
        _ = self.saver.save(sess, path)
        print("End model saved.")

    def restore(self, sess, path):
        self.saver.restore(sess, path)

