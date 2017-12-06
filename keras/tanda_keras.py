from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from utils import load_pretrained_tan


class TANDAImageDataGenerator(ImageDataGenerator):
    """Generate minibatches of image data with real-time data augmentation
       using a trained TAN.
    # Arguments
        tan: trained `TAN` object.
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first'
            mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 tan_path,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        super(TANDAImageDataGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format
        )
        self.tan = load_pretrained_tan(tan_path)
        self.session = K.get_session()

    def random_transform(self, x, seed=None):
        """Augment a single image tensor using a pretrained TAN. Still called
            `random_transform` for compatability reasons.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.
        # Returns
            A transformed version of the input (same shape).
        """
        if seed is not None:
            np.random.seed(seed)
        return self.tan.transform(self.session, x)
