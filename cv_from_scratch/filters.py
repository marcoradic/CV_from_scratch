import numpy as np
import scipy.ndimage
import scipy.signal


def convolve2d(input_image, kernel):
    return scipy.signal.convolve2d(input_image, kernel)


def convolve(input_signal, kernel):
    return np.convolve(input_signal, kernel)


def sobel(input_image, mode='both'):
    assert mode in ('both', 'x', 'y')
    horizontal_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    vertical_kernel = horizontal_kernel.T
    sobel_horizontal = convolve2d(
        input_image, horizontal_kernel / np.abs(horizontal_kernel).sum()
    )
    if mode == 'both':
        return convolve2d(sobel_horizontal, vertical_kernel / np.abs(vertical_kernel).sum())
    if mode == 'x':
        return sobel_horizontal
    if mode == 'y':
        return convolve2d(input_image, vertical_kernel / np.abs(vertical_kernel).sum())


def laplacian(input_image):
    # todo diagonal neighbour version
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return convolve2d(input_image.copy(), kernel / np.abs(kernel).sum())


def bilinear(input_image):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    return convolve2d(input_image.copy(), kernel / kernel.sum())


def box(input_image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size))
    return convolve2d(input_image.copy(), kernel / kernel.sum())


def gaussian_filter(input_image, sigma=2):
    """Convolves an image with a Gaussian

    Uses SciPy implementation for now
    
    Arguments:
        input_image {np.array} -- input to be convolved
        sigma {int} -- Gaussian Sigma param
    
    Returns:
        np.array -- Gaussian-convolved image
    """

    return scipy.ndimage.gaussian_filter(input_image.copy(), sigma)
