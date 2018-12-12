import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.misc


def imread(filepath):
    return np.asarray(Image.open(filepath))


def imshow(image, cmap="gray"):
    plt.imshow(image, cmap=cmap)
    plt.show()


def grayscale(image):
    return np.mean(image, axis=2)


def resize(image, size):
    """
    image : ndarray
    The array of image to be resized.

    size : int, float or tuple
    int - Percentage of current size.
    float - Fraction of current size.
    tuple - Size of the output image.
    
    Returns:
        ndarray -- resized image
    """
    return scipy.misc.imresize(image, size)
