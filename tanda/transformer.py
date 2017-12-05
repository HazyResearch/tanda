from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np

from skimage.util import crop, pad


class Transformer(object):

    def __init__(self, tfs):
        """Transforms data points given a set of transformation functions (TFs)

        @tfs: a list of TFs

        Note that each element of `tfs` will usually be a base TF with a
        specific parameter, e.g. `partial(TF_rotate, angle=2.5)`.
        """
        self.tfs = tfs
        # The number of available actions
        # Note this is synonymous with len(self.tfs) in the current
        # implementation, but might diverge in other ones, so we standardize
        self.n_actions = len(self.tfs)

    def pre_tf(self, x):
        return x

    def post_tf(self, x):
        return x

    def _apply(self, x, tf_seq, emit_incremental):
        """Apply a sequence of TFs to data point x
        
        @tf_seq: a list of indices referencing `self.tfs`
        @emit_incremental: If true, returns each incrementally-transformed
            image, _including_ the original image
        """
        # NOTE that we include the un-transformed datapoint as the first object!
        xcs = [self.post_tf(self.pre_tf(copy.deepcopy(x)))]

        # Apply the TFs, in the given order by default
        xc = self.pre_tf(copy.deepcopy(x))
        for i in tf_seq:
            xc = self.tfs[i](xc)
            xcs.append(self.post_tf(copy.deepcopy(xc)))

        # Return either just the final transformed version, or all the incremental ones
        if emit_incremental:
            return np.vstack(xcs)
        else:
            return xcs[-1]

    def transform(self, X, tf_seqs, emit_incremental=True):
        """Apply a sequence of TFs to a batch of data points X
        
        @tf_seqs: A matrix representing one list of indices, referencing
        `self.tfs`, for each data point x in X.
        @emit_incremental: If true, returns each incrementally-transformed
            image, _including_ the original image
        """
        xcs = [self._apply(x, t, emit_incremental) for x, t in zip(X, tf_seqs)]
        if emit_incremental:
            return np.array(xcs)
        else:
            return np.vstack(xcs)

    def random_transform(self, X, seq_len, emit_incremental=True, **kwargs):
        """Apply a random sequence of TFs to each x in X"""
        rand_seqs = np.random.randint(self.n_actions, size=(len(X), seq_len))
        return self.transform(
            X, rand_seqs, emit_incremental=emit_incremental, **kwargs
        )

    def transform_basic(self, X, train=False):
        return X


class ImageTransformer(Transformer):

    def __init__(self, tfs, dims):
        self.dims = dims
        self.size = np.prod(dims)
        super(ImageTransformer, self).__init__(tfs)

    def pre_tf(self, img):
        return np.reshape(img, self.dims)

    def post_tf(self, img):
        return np.reshape(img, [self.size])


class PadCropTransformer(ImageTransformer):
    """Pad and then (randomly) crop back to same original size."""
    def __init__(self, tfs, dims, pad_px=4, pad_mode='edge'):
        self.pad_px  = pad_px
        self.pad_mode = pad_mode
        super(PadCropTransformer, self).__init__(tfs, dims)

    def transform_basic(self, X, train=False):
        return np.vstack([
            self.post_tf(self.pre_tf(copy.deepcopy(X[i])), train=train)
            for i in range(X.shape[0])
        ])

    def pre_tf(self, img):
        img = np.reshape(img, self.dims)

        # Pad image by n_pixels on each side
        return pad(
            img,
            [(self.pad_px, self.pad_px) for _ in range(2)] + [(0,0)],
            mode=self.pad_mode
        )

    def post_tf(self, img, train=True):
        """Note that we assume a square image and crop centered if not
        training, randomly if training.
        """
        assert self.dims[0] == self.dims[1]
        if train:
            # Take a random crop of the original size
            crop_sizes = np.random.randint(0, 2*self.pad_px+1, [2])
            crops = [(c, 2*self.pad_px-c) for c in crop_sizes]
        else:
            crops = [(self.pad_px, self.pad_px) for _ in range(2)]

        # For channel dimension don't do any cropping
        crops += [(0,0)]
        return np.reshape(crop(img, crops, copy=True), [self.size])
