import numpy as np

from . import filters
from . import utils


def histogram_equalization(input_image):
    """Full histogram equalization
    
    Arguments:
        input_image {[type]} -- [description]
    """
    prefiltered = filters.gaussian_filter(input_image)
    histogram = np.histogram(prefiltered, bins=256)
    histogram_cdf = np.cumsum(histogram[0])
    histogram_cdf_scaled = histogram_cdf * 256.0 / histogram_cdf.max()
    lookup = lambda x: histogram_cdf_scaled[int(x)]
    result = np.zeros_like(prefiltered)
    for i in range(prefiltered.shape[0]):
        for j in range(prefiltered.shape[1]):
            result[i, j] = lookup(prefiltered[i, j])
    return result


def threshold(input_image, threshold):
    """
    Binarizes an image based on a threshold

    pixel values larger than the threshold will be assigned a value of 255, smaller a value of 0
    """
    if threshold > 255 or threshold < 0:
        raise Exception("Threshold value not valid")
    thresholded_image = input_image.copy()
    thresholded_image[thresholded_image >= threshold] = 255
    thresholded_image[thresholded_image < threshold] = 0
    return thresholded_image


def inverse(input_image):
    return input_image.max() - input_image

def normalize(input_image):
    return input_image / 255.


def gaussian_pyramid(input_image, max_layer=4, downscale=2.0):
    pyramid = [input_image]
    for _ in range(max_layer):
        layer_image = filters.gaussian_filter(pyramid[-1])
        layer_image = utils.resize(layer_image, 1 / downscale)
        pyramid.append(layer_image)
    return pyramid


def laplacian_pyramid(input_image, max_layer=4, downscale=2.0):
    layer_image = input_image
    layer_difference = layer_image - filters.gaussian_filter(layer_image)
    difference_pyramid = [layer_difference]
    for _ in range(max_layer):
        layer_image = utils.resize(layer_image, 1 / downscale)
        layer_difference = layer_image - filters.gaussian_filter(layer_image)
        layer_image = filters.gaussian_filter(layer_image)
        difference_pyramid.append(layer_difference)
    return (difference_pyramid, layer_image)
