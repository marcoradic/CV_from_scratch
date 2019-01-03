import sys
sys.path.append('..')

import cv_from_scratch
from cv_from_scratch import transformations, filters, utils
import numpy as np
import numba
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

import glob


def hough(image, num_lines=2):
    gradient_mag = image  # np.abs(filters.sobel(image))
    angles = np.deg2rad(np.linspace(-90.0, 90.0, 180))
    h, w = image.shape
    diagonal = np.ceil(np.sqrt(w * w + h * h))
    distances = np.linspace(-diagonal, diagonal, 2 * diagonal)
    cos = np.cos(angles)
    sin = np.sin(angles)
    accumulator = np.zeros((int(2 * diagonal), len(angles)), dtype=np.uint64)
    utils.imshow(gradient_mag)
    # p_x, p_y = np.argwhere(gradient_mag > 1.)[:, 0], np.argwhere(gradient_mag > 1.)[:, 1]
    p_x, p_y = np.nonzero(gradient_mag)

    for x, y in tqdm(zip(p_x, p_y)):
        for angle in range(len(angles)):
            distance = x * cos[angle] + y * sin[angle]
            index = int(np.digitize(distance, distances))
            try:
                accumulator[index, angle] += 1
            except IndexError:
                pass
    plt.imshow(
        accumulator,
        cmap="jet",
        extent=[-90, 90, distances[-1], distances[0]],
        aspect=0.5,
    )
    plt.show()

    return accumulator, angles, distances


image = np.zeros((50, 50))
image[30:50, 0:20] = np.eye(20)
image[30:50, 30:50] = np.rot90(np.eye(20))
# image[20:40, 20:40] = np.eye(20) + np.rot90(np.eye(20))
# image = utils.grayscale(utils.imread('lanes.png'))
accumulator, thetas, rhos = hough(image)

threshold = 14
idxs = np.argwhere(accumulator > threshold)
print(idxs)
print(np.argmax(accumulator))
ax = plt.subplot()
ax.imshow(image)
ax.set_ylim(0, image.shape[0])
ax.set_xlim(0, image.shape[1])
for idx in idxs:
    x, y = idx
    dist = rhos[int(x)]
    angle = thetas[y]
    y0 = dist / np.sin(angle)
    y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
    ax.plot([y0, y1], [0, image.shape[0]], c="r")
    print(f"rho {dist}, theta {np.rad2deg(angle)}")
plt.show()
