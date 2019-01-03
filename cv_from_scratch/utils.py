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


def timer(f):
    """
    Decorator function that times the execution time of a decorated function
    """
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        print("{0} took {1:.2f}s".format(f.__qualname__, time.time() - start))
        return result

    return wrapper
