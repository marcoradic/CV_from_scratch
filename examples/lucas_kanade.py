import sys
sys.path.append('..')

from cv_from_scratch import transformations, filters, utils
import numpy as np
import numba
import imageio
import matplotlib.pyplot as plt

import glob


@utils.timer
@numba.jit
def optical_flow(image1, image2):
    """Lucas-Kanade optical Flow algorithm with hard box window
    
    Arguments:
        image1 {np.array} -- first input image
        image2 {np.array} -- second input image, frame following the first image as reference
    
    Returns:
        u, v, eigsum, eigprod - u and v result arrays which describe the optical flow, additionally arrays of eigenvalues
    """

    # normalize to range [0, 1]
    image1 = transformations.normalize(image1)
    image2 = transformations.normalize(image2)
    # gradients
    x_d = filters.sobel(image1, mode="x")
    y_d = filters.sobel(image1, mode="y")
    t_d = image1 - image2
    u, v = np.zeros_like(image1), np.zeros_like(image2)
    # window size parameter
    l = 8
    # collect sum and product of eigenvalues from structure tensor per pixel
    eigsum = np.zeros_like(image1)
    eigprod = np.zeros_like(image1)
    for y in range(l, image1.shape[1] - l):
        for x in range(l, image1.shape[0] - l):
            A = np.zeros((2, 2)) # structure tensor
            x_window = x_d[x - l : x + l + 1, y - l : y + l + 1]
            y_window = y_d[x - l : x + l + 1, y - l : y + l + 1]
            t_window = t_d[x - l : x + l + 1, y - l : y + l + 1]
            A[0, 0] = np.sum(x_window * x_window)
            A[0, 1] = np.sum(x_window * y_window)
            A[1, 0] = A[0, 1]
            A[1, 1] = np.sum(y_window * y_window)
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[0, 1]
            eigsum[x, y] = A[0, 0] + A[1, 1]
            eigprod[x, y] = A[0, 0] * A[1, 1]
            if A[0, 0] + A[1, 1] < 0.01: # 2 vanishing eigenvalues, so no information
                continue
            elif det <= 1e-3: # one vanishing eigenvalue -> edge
                continue
            b_0 = -np.sum(x_window * t_window)
            b_1 = -np.sum(y_window * t_window)
            u[x, y] = (A[1, 1] * b_0 - A[0, 1] * b_1) / det
            v[x, y] = (-A[0, 1] * b_0 + A[0, 0] * b_1) / det
    return u, v, eigsum, eigprod


def process_images(images):
    """Processes an image list, saves the result as a gif
    
    Arguments:
        images {list} -- list of image np.array s
    """

    results = []
    eigsums, eigprods = [], []
    for i in range(len(images) - 1):
        print(f"processing image {i} of {len(images)-1}")
        u, v, eigsum, eigprod = optical_flow(images[i], images[i + 1])
        results.append(u ** 2 + v ** 2)
        eigsums.append(eigsum ** 2)
        eigprods.append(eigprod)
    results = np.array(results)
    imageio.mimsave("taxi.gif", results, duration=0.1)
    """ uncomment to save gifs of sum and product of eigenvalues 
    eigsums = np.array(eigsums)
    imageio.mimsave("eigsums.gif", eigsums, duration=0.1)
    eigprods = np.array(eigprods)
    imageio.mimsave("eigprods.gif", eigprods, duration=0.1)
    """


if __name__ == "__main__":
    folder = "taxi"
    images = glob.glob(f"{folder}/*")
    images = sorted(images)
    images = [utils.imread(im) for im in images]
    process_images(images)
