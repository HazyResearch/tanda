from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from PIL import Image, ImageEnhance
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage import img_as_ubyte, img_as_float
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import adjust_gamma
from skimage.filters import gaussian
from skimage.morphology import dilation, erosion
from skimage.transform import rotate, rescale, swirl, warp, AffineTransform
from skimage.util import random_noise, pad, crop


def np_to_pil(x):
    """Converts from np image in skimage float format to PIL.Image"""
    x = np.squeeze(np.uint8(img_as_ubyte(x)))
    return Image.fromarray(np.uint8(img_as_ubyte(x)))


def pil_to_np(x):
    """Converts from PIL.Image to np float format image"""
    x = np.asarray(x)
    if len(x.shape) == 2: 
        x = x[:,:,np.newaxis]
    return img_as_float(np.asarray(x))


def TF_crop_pad_flip(x, n_pixels=4, pad_mode='edge'):
    """Replicate the augmentation used in ResNet paper."""
    if np.random.random() < 0.5:
        x = TF_horizontal_flip(x)
    return TF_crop_pad(x, n_pixels=n_pixels, pad_mode=pad_mode)


def TF_crop_pad(x, n_pixels=4, pad_mode='edge'):
    """Pad image by n_pixels on each size, then take random crop of same
    original size.
    """
    assert len(x.shape) == 3
    h, w, nc = x.shape

    # First pad image by n_pixels on each side
    padded = pad(x, [(n_pixels, n_pixels) for _ in range(2)] + [(0,0)],
        mode=pad_mode)

    # Then take a random crop of the original size
    crops = [(c, 2*n_pixels-c) for c in np.random.randint(0, 2*n_pixels+1, [2])]
    # For channel dimension don't do any cropping
    crops += [(0,0)]
    return crop(padded, crops, copy=True)


def TF_horizontal_flip(x):
    return np.fliplr(x)


def TF_shift_hue(x, shift=0.0):
    assert len(x.shape) == 3
    h, w, nc = x.shape

    hsv = rgb2hsv(x)
    hsv[:,:,0] += shift
    return hsv2rgb(hsv)


def TF_enhance_contrast(x, p=1.0, target=None):
    assert len(x.shape) == 3
    h, w, nc = x.shape

    enhancer = ImageEnhance.Contrast(np_to_pil(x))
    return pil_to_np(enhancer.enhance(p))


def TF_enhance_brightness(x, p=1.0):
    assert len(x.shape) == 3
    h, w, nc = x.shape

    enhancer = ImageEnhance.Brightness(np_to_pil(x))
    return pil_to_np(enhancer.enhance(p))


def TF_enhance_color(x, p=1.0):
    assert len(x.shape) == 3
    h, w, nc = x.shape

    enhancer = ImageEnhance.Color(np_to_pil(x))
    return pil_to_np(enhancer.enhance(p))


def TF_enhance_sharpness(x, p=1.0):
    assert len(x.shape) == 3
    h, w, nc = x.shape

    enhancer = ImageEnhance.Sharpness(np_to_pil(x))
    return pil_to_np(enhancer.enhance(p))


def TF_adjust_gamma(x, gamma=1.0):
    assert len(x.shape) == 3
    h, w, nc = x.shape
    return adjust_gamma(x, gamma)


def TF_rotate(x, angle=0.0, target=None):
    assert len(x.shape) == 3
    h, w, nc = x.shape

    # Rotate using edge fill mode
    return rotate(x, angle, mode='edge', order=1)


def TF_zoom(x, scale=1.0, target=None):
    assert len(x.shape) == 3
    h, w, nc = x.shape
    assert h == w

    # Zoom
    xc   = rescale(x, scale)
    diff = h - xc.shape[0]
    d    = int(np.floor(diff / 2.0))
    if d >= 0:
        padding = ((d, d),(d, d),(0,0))
        if diff % 2 != 0:
            padding = ((d,d+1), (d,d + 1),(0,0))
        return np.pad(xc, padding, mode='edge')
    else:
        return xc[-d:h-d, -d:w-d].reshape(h, w, nc)


def TF_noise(x, magnitude=None, mode='gaussian', mean=0.0):
    if mode in ['gaussian', 'speckle']:
        return random_noise(x, mode=mode, mean=mean, var=magnitude)
    elif mode in ['salt', 'pepper', 's&p']:
        return random_noise(x, mode=mode, amount=magnitude)
    else:
        return random_noise(x, mode=mode)


def TF_blur(x, sigma=0.0, target=None):
    return gaussian(x, sigma=sigma, multichannel=True)


def TF_shear(x, shear=0.0):
    assert len(x.shape) == 3
    h, w, nc = x.shape

    # Perform shear
    xc = warp(x, AffineTransform(shear=shear), mode='edge')
    return xc


def TF_swirl(x, strength=0.0, radius=100):
    assert len(x.shape) == 3
    h, w, nc = x.shape
    
    # Perform swirl 
    return swirl(x, strength=strength, radius=radius, mode='edge')


def TF_elastic_deform(img, alpha=1.0, sigma=1.0):
    """Elastic deformation of images as described in Simard 2003"""
    assert len(img.shape) == 3
    h, w, nc = img.shape
    if nc != 1:
        raise NotImplementedError("Multi-channel not implemented.")

    # Generate uniformly random displacement vectors, then convolve with gaussian kernel
    # and finally multiply by a magnitude coefficient alpha
    dx = alpha * gaussian_filter(
        (np.random.random((h, w)) * 2 - 1), sigma, mode="constant", cval=0
    )
    dy = alpha * gaussian_filter(
        (np.random.random((h, w)) * 2 - 1), sigma, mode="constant", cval=0
    )

    # Map image to the deformation mesh
    x, y    = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    return map_coordinates(img.reshape((h,w)), indices, order=1).reshape(h,w,nc)


def TF_flip(img, flip_type=0): 
    """perform horizontal or vertical flip"""
    assert len(img.shape) == 3 

    flips = ['null', 'horizontal', 'vertical']
    if flips[flip_type] == 'null':
        return img
    elif flips[flip_type] == 'horizontal': 
        return np.fliplr(img)
    else: 
        return np.flipud(img) 


def TF_jitter(img, ps): 
    """does a jitter on the image, i.e. shifts the pixels of image on 
    some axis by randomly chosen factor""" 
    ox, oy = ps
    return np.roll(np.roll(img, ox, -1), oy, -2)


def TF_power(img, pow_std=0.0): 
    """raises original image to some power"""
    power = np.random.randn() * pow_std + 1.0
    pow_im = np.sign(img) * (np.absolute(img)**power)
    return pow_im 


def TF_dilation(img):
    return dilation(img)


def TF_erosion(img):
    return erosion(img)


def TF_translate(img, x, y):
    return warp(img, AffineTransform(translation=(x, y)), mode='edge')
