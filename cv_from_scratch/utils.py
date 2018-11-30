import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def imread(filepath):
    return np.asarray(Image.open(filepath))


def imshow(image, cmap="gray"):
    plt.imshow(image, cmap=cmap)
    plt.show()


def grayscale(image):
    return np.mean(image, axis=2)
