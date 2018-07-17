import os
import cv2
import numpy as np
from keras.datasets import mnist


def load_mnist():
    (x, _), (_, _) = mnist.load_data()
    x = x / 255.
    return x.reshape((-1, 1, 28, 28))


def save_projection_image(memory, lattice_size):
    width = memory.shape[2]
    height = memory.shape[1]
    proj_map = np.zeros((lattice_size[0] * height, lattice_size[1] * width))
    c = 0
    for i in range(lattice_size[0]):
        for j in range(lattice_size[1]):
            proj_map[
                i * height:(i + 1) * height, j * width:(j + 1) * width] = memory[c]
            c += 1
    if not os.path.exists('backprojections'):
        os.mkdir('backprojections')
    cv2.imwrite('backprojections/DNM_projection.png', 255 * proj_map)
