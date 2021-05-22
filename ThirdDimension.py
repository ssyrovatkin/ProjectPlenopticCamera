from skimage import io
import numpy as np
import matplotlib.pyplot as plt

import imageio
import skimage
from pathlib import Path


class ThirdDimension:
    def __init__(self, file, destination=None, presentation_mode=False):
        self.presentation = presentation_mode

        image = skimage.io.imread(file)
        image = image / image.max()

        name = Path(file).stem
        if destination is not None:
            self.output_path = Path(destination) / name
        else:
            self.output_path = name

        if self.presentation:
            plt.figure(figsize=(16, 16))
            plt.imshow(image)
            plt.savefig('merc1.png')

        images_3d = []

        for parallax in range(-10, 0, 1):
            image_l = image[:, 0:-abs(parallax), :]
            image_r = image[:, abs(parallax):, :]

            if parallax <= 0:
                image_l, image_r = image_r, image_l
            mercedes3d = np.zeros_like(image_l)
            mercedes3d[:, :, 0] = image_r[:, :, 0]
            mercedes3d[:, :, 1] = image_l[:, :, 1]
            mercedes3d[:, :, 2] = image_r[:, :, 2]

            images_3d.append(mercedes3d)

        from copy import copy

        result = copy(images_3d)
        images_3d.reverse()
        result = result + copy(images_3d)

        plt.figure(figsize=(16, 16))
        plt.imshow(result[5])

        output_file = 'example.gif'
        duration = 0.1
        imageio.mimsave(output_file, result, duration=duration)



