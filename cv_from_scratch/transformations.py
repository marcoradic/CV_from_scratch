import numpy as np

from . import filters


def histogram_equalization(input_image):
    """Full histogram equalization
    
    Arguments:
        input_image {[type]} -- [description]
    """
    prefiltered = filters.gaussian_filter(input_image)
    histogram = np.histogram(prefiltered, bins=255)
    histogram_cdf = np.cumsum(histogram)
    histogram_cdf_scaled = histogram_cdf * 255.0 / histogram_cdf.max()
    lookup = lambda x: histogram_cdf_scaled[min(254, int(x))]
    result = np.zeros_like(prefiltered)
    for i in range(prefiltered.shape[0]):
        for j in range(prefiltered.shape[1]):
            result[i, j] = lookup(prefiltered[i, j])
        
def threshold(input_image, threshold):
    pass

