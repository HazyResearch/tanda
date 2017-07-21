import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from tanda.discriminator import Discriminator


def generate_data(n, d=2, r=1, l=1, inside_ball=True):
    """Generate n points uniformly, in d dimensions, in a ball of radius r."""
    # Use simple rejection sampling, from uniform dist. over hypercvube of dim.
    pts = []
    while len(pts) < n:
        x = l * (2 * np.random.random(d) - 1)
        xr = np.linalg.norm(x)
        if (inside_ball and xr <= r) or (not inside_ball and xr > r):
            pts.append(x)
    pts = np.vstack(pts)
    return pts


class OracleDiscriminator(Discriminator):
    """Return the oracle answer"""
    def get_logits_op(self, x_input, r=1.0, **kwargs):
        """We just check how far outside the radius it is"""
        x_norm = tf.sqrt(tf.reduce_sum(tf.pow(x_input, 2), 1))
        return tf.reshape(tf.cast(tf.less(x_norm, 1.0), tf.float32), [-1, 1])


def plot_synthetic(X_in, X_t, pred_in=None, pred_t=None, title=None,
    show_plot=True, savepath=None, r=1.0):
    # If no predictions from the discriminator are provided, just fill in ones
    pred_in = pred_in if pred_in is not None else np.ones(X_in.shape[0])
    pred_t = pred_t if pred_t is not None else np.ones(X_t.shape[0])

    # Plot the true decision boundary, i.e. the circle with radius r
    thetas = np.arange(0, 2*np.pi, 0.01*np.pi)

    s = 1.0 * r
    plt.plot(s * np.cos(thetas), s * np.sin(thetas), r'--', color='#810f7c')

    # We split up into four groups by in vs. t, predicted in vs. t
    X1 = X_in[pred_in == 1]
    if X1.shape[0] > 0:
        plt.scatter(X1[:,0], X1[:,1], marker=r'o', color='#008fd5')

    X2 = X_in[pred_in == 0]
    if X2.shape[0] > 0:
        plt.scatter(X2[:,0], X2[:,1], marker=r'X', color='#008fd5')

    X3 = X_t[pred_t == 1]
    if X3.shape[0] > 0:
        plt.scatter(X3[:,0], X3[:,1], marker=r'o', color='#fc4f30')

    X4 = X_t[pred_t == 0]
    if X4.shape[0] > 0:
        plt.scatter(X4[:,0], X4[:,1], marker=r'X', color='#fc4f30')

    # Other formatting
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])

    # Build and show plot
    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath)
    if show_plot:
        plt.show()
    plt.close()


def save_data_plot(session, tan, X, batch_size, savepath):
    n = X.shape[0]
    X_t, pred_in, pred_t = [], [], []
    for i in range(0, n, batch_size):
        X_b = X[i : i+batch_size, :]
        X_t_b, pred_in_b, pred_t_b = tan.get_transformed_data_and_predictions(
            session, X_b)
        X_t.append(X_t_b)
        pred_in.append(pred_in_b)
        pred_t.append(pred_t_b)
    plot_synthetic(X, np.vstack(X_t), pred_in=np.concatenate(pred_in),
        pred_t=np.concatenate(pred_t), show_plot=False, savepath=savepath)
