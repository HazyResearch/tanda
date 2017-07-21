import tensorflow as tf

from discriminator import Discriminator


def nnet(input_tensor, n_hidden=4):
    h = tf.layers.dense(input_tensor, n_hidden,
        activation=tf.nn.sigmoid, name='h_0')
    return tf.layers.dense(h, 1, name='h_1')


class SimpleDiscriminator(Discriminator):
    """A simple two-layer neural net"""
    def get_logits_op(self, x_input, **kwargs):
        """Returns logits"""
        return nnet(x_input)
