from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from .discriminator import DCNN


ADAM = tf.train.AdamOptimizer
SGD  = tf.train.GradientDescentOptimizer


def get_mse_loss(mse, mse_term, eps=1e-6, mean=False):
    z = mse_term * (1.0 / (mse + eps))
    return tf.reduce_mean(z) if mean else z


def per_image_std_map(X, img_dims, dims_out):
    imgs = tf.reshape(X, [-1] + img_dims)
    imgs = tf.map_fn(tf.image.per_image_standardization, imgs)
    return tf.reshape(imgs, dims_out)


class TFQ(object):
    def __init__(self, generator, size):
        self.generator = generator
        self.size = size
        self._q = None
        self._i = size

    def init_q(self, session):
        self._q = self.generator.get_action_sequence(session, self.size)
        self._i = 0

    def next(self, session):
        if self._i >= self.size:
            self.init_q(session)
        self._i += 1
        return self._q[self._i - 1, :]


class TAN(object):
    """Transormation Adversarial Network"""
    def __init__(self, discriminator, generator, transformer, d_lr, g_lr,
                 mse_term=1.0, mse_layer=None, d_trainer=ADAM, g_trainer=SGD,
                 reuse=False, gamma=0.0, per_img_std=False, train_disc=True,
                 tf_seq_queue_size=None):
        self.discriminator = discriminator
        self.generator = generator
        self.transformer = transformer
        # Note that we inherit the data dimensions from the discriminator
        self.dims = self.discriminator.dims
        self.d    = np.prod(self.dims) # Flattened size
        # We can optionally not train the disc e.g. if using an oracle disc
        self.train_disc = train_disc
        # Build training operations
        self.d_train_op = None
        self.g_train_op = None
        self.batch_size = self.generator.batch_size
        self.reuse      = reuse
        # Optionally initialize a TF seq queue for batch generation
        if not tf_seq_queue_size:
            self.tf_q = None
        else:
            self.tf_q = TFQ(generator, tf_seq_queue_size)
        # Build model graph
        self._build(d_lr, g_lr, mse_term, mse_layer, d_trainer, g_trainer,
            gamma, per_img_std)

    def _build(self, d_lr, g_lr, mse_term, mse_layer, d_trainer, g_trainer,
        gamma, per_img_std):
        """Build the TAN computation graph"""
        T = self.generator.seq_len

        # Placeholders for basic input data
        self.data  = tf.placeholder(tf.float32, (None, self.d))
        batch_size = tf.shape(self.data)[0]

        # For each datapoint we expect the *original data point first*,
        # then the T = seq_len incremental transformed versions
        self.transformed_data = tf.placeholder(tf.float32, (None, T+1, self.d))
        
        ###
        ### DISCRIMINATOR LOSS
        ###
        # Get discriminator logits over real data
        with tf.variable_scope("discriminator", reuse=self.reuse):
            D_real = self.discriminator.get_logits_op(self.data, 
                per_img_std=per_img_std, get_layers=(mse_layer is not None))

        # Separate layers from loss, or use pixels as layers
        if mse_layer is not None:
            D_real, D_real_layers = D_real
            data = D_real_layers[-(mse_layer + 1)]
        else:
            if per_img_std:
                data = per_image_std_map(self.data, self.dims,
                    [batch_size, self.d])
                data_t = per_image_std_map(self.transformed_data, self.dims,
                    [batch_size, T+1, self.d])
            else:
                data, data_t = self.data, self.transformed_data

        # Get discriminator logits over *final* transform data
        data_tf = self.transformed_data[:, -1, :]
        with tf.variable_scope("discriminator", reuse=True):
            D_tf = self.discriminator.get_logits_op(data_tf,
                per_img_std=per_img_std, get_layers=False)

        # Define discriminative loss
        real_loss   = self.discriminator.get_loss_op(D_real)
        tf_loss     = self.discriminator.get_loss_op(D_tf, positive=False)
        self.D_loss = 0.5 * (real_loss + tf_loss)

        ###
        ### GENERATOR LOSS
        ###
        # Get the logits for each incrementally-transformed datapoint
        with tf.variable_scope("discriminator", reuse=True):
            D_tf_g_array = tf.TensorArray(tf.float32, T + 1)
            if mse_layer is not None:
                data_t_array = tf.TensorArray(tf.float32, T + 1)
            for i in range(T + 1):
                # Note: Here we pass in train=False, which is passed to e.g.
                # batch_norm and other operators in the discriminator that are
                # stochastic during training
                d_tf_g = self.discriminator.get_logits_op(
                    self.transformed_data[:, i, :], per_img_std=per_img_std,
                    train=False, get_layers=(mse_layer is not None))
                # Separate loss and layer
                if mse_layer is not None:
                    d_tf_g, d_tf_g_layers = d_tf_g
                    # Use negative index to retrieve layer to use
                    d_tf_g_layer = d_tf_g_layers[-(mse_layer + 1)]
                    data_t_array = data_t_array.write(i, d_tf_g_layer)
                D_tf_g_array = D_tf_g_array.write(i, d_tf_g)
            # D_tf_g is reshaped to [batch_size, T+1, dim]
            D_tf_g = tf.transpose(D_tf_g_array.stack(), perm=[1, 0, 2])
            if mse_layer is not None:
                data_t = tf.transpose(data_t_array.stack(), perm=[1, 0, 2])
        
        # Define generative loss for training
        # G_loss_all is a batch_size x (T+1) matrix with the discriminator
        # losses of all the incremental transformed data points
        G_loss_all = self.discriminator.get_loss_op(D_tf_g, mean=False)

        # Add MSE term to generator objective function here
        shape  = [batch_size, 1, tf.shape(data_t)[2]]
        data_r = tf.tile(tf.reshape(data, shape), [1, T+1, 1])
        mse    = tf.reduce_mean(tf.square(data_r - data_t), 2)
        G_loss_all = G_loss_all + get_mse_loss(mse, mse_term)
        
        # Get the change in loss between each incremental transformation
        # G_loss_deltas is a [batch_size, T] Tensor
        self.G_loss_deltas = G_loss_all[:, 1:] - G_loss_all[:, :-1]
        
        # Get the policy loss op
        q = self.generator.get_policy_loss_op(self.G_loss_deltas, gamma)

        ###
        ### TRAINING OPS
        ###

        # Define discriminative operation
        self.d_train_op = None
        if self.train_disc:
            d_name = "discriminator"
            d_vars = [
                v for v in tf.trainable_variables() if v.name.startswith(d_name)
            ]
            d_update_ops = [
                u for u in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if u.name.startswith(d_name)
            ]
            with tf.variable_scope(d_name, reuse=self.reuse):
                d_step = tf.Variable(0, trainable=False)

                # Note: This is necessary for batch_norm to be handled correctly
                with tf.control_dependencies(d_update_ops):
                    self.d_train_op = d_trainer(d_lr).minimize(self.D_loss,
                        global_step=d_step, var_list=d_vars)

        # Define generative operation
        g_vars = [
            v for v in tf.trainable_variables()
            if v.name.startswith(self.generator.name)
        ]
        g_update_ops = [
            u for u in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if u.name.startswith(self.generator.name)
        ]
        with tf.variable_scope(self.generator.name, reuse=self.reuse):
            g_step = tf.Variable(0, trainable=False)

            # Note: This is necessary for batch_norm to be handled correctly
            with tf.control_dependencies(g_update_ops):
                self.g_train_op = g_trainer(g_lr).minimize(q, 
                    global_step=g_step, var_list=g_vars)

        # Define predictions
        self.d_pred = tf.to_int32(tf.greater(tf.sigmoid(D_real), 0.5))
        self.g_pred = tf.to_int32(tf.greater(tf.sigmoid(D_tf), 0.5))

        ###
        ### LOGGING
        ###

        # Create summary for D_loss
        D_loss_summary = tf.summary.scalar("disc_loss", self.D_loss)
        # Get discriminative loss over transformed image for generator loss
        # Note a squeeze is performed automatically in the slice for MSE
        msef = tf.reduce_mean(tf.square(data_r[:, -1, :] - data_t[:, -1, :]), 1)
        self.G_loss     = self.discriminator.get_loss_op(D_tf)
        self.G_loss_mse = self.G_loss + get_mse_loss(msef, mse_term, mean=True)
        # Create summaries for generator loss
        G_loss_summary     = tf.summary.scalar("gen_loss", self.G_loss)
        G_loss_mse_summary = tf.summary.scalar("gen_mse_loss", self.G_loss_mse)
        # Create alias summaries for random loss
        self.R_loss        = self.G_loss
        self.R_loss_mse    = self.G_loss_mse
        R_loss_summary     = tf.summary.scalar("rand_loss", self.R_loss)
        R_loss_mse_summary = tf.summary.scalar("rand_mse_loss", self.R_loss_mse)
        # Merge summaries
        dg_summaries      = [D_loss_summary, G_loss_summary, G_loss_mse_summary]
        self.dg_summary   = tf.summary.merge(dg_summaries)
        r_summaries       = [R_loss_summary, R_loss_mse_summary] 
        self.rand_summary = tf.summary.merge(r_summaries)

        ### Saver
        # NOTE: We only save the generative model, this way compatible with a
        # larger range of end models
        vars_list = [
            v for v in tf.trainable_variables()
            if v.name.startswith(self.generator.name)
        ]
        self.saver = tf.train.Saver(var_list=vars_list)

    def get_transformed_data(self, session, data, emit_incremental=False,
        n_seqs_per_example=1):
        """Transform data
            @session: a TensorFlow session
            @data:    original training data batch
            @emit_incremental: return incrementally transformed data points?
            @n_seqs_per_example: number of sampled transformation sequences
                applied to each data point
        Returns a tuple of transformed data, the sequences applied, and the
            original data repeated n_seqs_per_example times
        """
        # Replicate n_seqs_per_example times
        data_rep = np.tile(data, (n_seqs_per_example, 1))
        # Get action sequences
        tf_seqs = self.generator.get_action_sequence(session, data_rep.shape[0])
        # Transform data
        return (self.transformer.transform(
            data_rep, tf_seqs, emit_incremental=emit_incremental
        ), tf_seqs, data_rep)

    def transform(self, session, x):
        """Transform single data point
            @session:   a TensorFlow session
            @x:         original training data point
        Returns a transformed data point; uses TF queue if initialized
        """
        # Get action sequences
        if self.tf_q is not None:
            tf_seq = self.tf_q.next(session)
        else:
            tf_seq = self.generator.get_action_sequence(session, 1)[0, :]
        # Transform data
        return self.transformer(x, tf_seq)

    def get_random_loss(self, session, data, gen_loss=None):
        """
        Return loss with random transformations; if gen_loss provided
        as scalar value, return sumary for gen_loss / rand_loss as well.
        """
        # Make sure is in proper dims
        r_data_b = self.transformer.transform_basic(data)
        # Get random sequence of transformations
        seq_len = self.generator.seq_len
        # Get both incremental and final transformed data points
        rand_tf_data_inc = self.transformer.random_transform(
            data, seq_len, emit_incremental=True
        )
        loss, summary = session.run([self.R_loss, self.rand_summary], {
            self.data: r_data_b,
            self.transformed_data: rand_tf_data_inc,
        })
        if gen_loss is not None:
            ratio_val = tf.summary.Summary.Value(
                tag="gen_rand_loss_ratio",
                simple_value=float(loss) / gen_loss
            )
            ratio_summary = tf.summary.Summary(value=[ratio_val])
            return loss, summary, ratio_summary
        else:
            return loss, summary

    def train_step(self, session, data, n_disc_steps, n_gen_steps, n_sample=1):
        # Optionally transform test data (to match transformed data)
        # E.g. see PadCropTransformer class
        d_data_test = self.transformer.transform_basic(data)
        # Update discriminator
        for _ in range(n_disc_steps):
            # Get transformed data
            tf_d_data, _, _ = self.get_transformed_data(session, data,
                emit_incremental=True)
            # Get loss and if trainind discriminator (default) execute train op
            fd = {self.data: d_data_test, self.transformed_data: tf_d_data}
            if self.train_disc:
                d_loss, _ = session.run([self.D_loss, self.d_train_op], fd)
            else:
                d_loss = session.run([self.D_loss], fd)

        # Update generator
        for _ in range(n_gen_steps):            
            # Get both incrementally-transformed data and final version
            tf_g_data_inc, tf_seqs, data_rep = self.get_transformed_data(
                session, data, emit_incremental=True,
                n_seqs_per_example=n_sample
            )
            # Get the feed_dict for the training step
            # Note that the get_feed method will make sure that the action
            # sequences sampled are the same as the ones used to generate the
            # transformed data above!
            g_feed = self.generator.get_feed(tf_seqs)
            g_feed.update({self.transformed_data: tf_g_data_inc})
            # Optionally transform test data (to match transformed data)
            # E.g. see PadCropTransformer class
            data_rep = self.transformer.transform_basic(data_rep)
            g_feed.update({self.data: data_rep})
            # Define training op
            g_loss, summary, _, g_loss_deltas = session.run(
                [self.G_loss, self.dg_summary, self.g_train_op,
                 self.G_loss_deltas], g_feed
            )
        # Return losses
        return d_loss, g_loss, summary, g_loss_deltas, tf_seqs

    def get_transformed_data_and_predictions(self, session, data):
        tf_data, _, _  = self.get_transformed_data(session, data, True)
        d_y, g_y    = session.run([self.d_pred, self.g_pred], {
            self.data: data, self.transformed_data: tf_data    
        })
        final_tf_data = np.squeeze(tf_data[:, -1, :])
        return final_tf_data, np.ravel(d_y), np.ravel(g_y)

    def save(self, session, save_path):
        _ = self.saver.save(session, save_path)

    def restore(self, session, save_path):
        self.saver.restore(session, save_path)


def PretrainedTAN(G, T, dims, session, checkpoint_path, tf_seq_queue_size=5000):
    # Build dummy discriminator
    D = DCNN(dims=dims)
    # Build TAN
    tan = TAN(D, G, T, 0, 0, tf_seq_queue_size=tf_seq_queue_size)
    tan.restore(session, checkpoint_path)
    return tan
