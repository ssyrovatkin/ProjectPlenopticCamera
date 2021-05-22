import numpy as np
import cv2
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

import imageio
import skimage
from pathlib import Path

second_row_shift = 0
first_row_shift = 5


class Parallax:

    def __init__(self, file, destination=None, presentation_mode=False):
        """
        ЗДЕСЬ БУДЕТ ОПИСАНИЕ КЛАССА
        :param file: str, path to input image
        """

        name = Path(file).stem
        if destination is not None:
            self.output_path = Path(destination) / name
        else:
            self.output_path = name
        image = skimage.io.imread(file)
        image = image / image.max()
        image_array = np.array(image)

        self.h = image.shape[0]
        self.w = image.shape[1]

        self.presentation = presentation_mode

        if self.presentation:
            self.save(image_array, 'original.png')  # cmap='gray'

            self.save(image_array[int(0.30*self.h):int(0.34*self.h), int(0.64*self.w):int(0.67*self.w)], 'zoomed.png')  # cmap='gray'

        mask_B, mask_G1, mask_G2, mask_R = self.get_bayer_pattern_masks()

        image_interpolated = self.high_quality_bayer_interpolation(image_array)

        if self.presentation:
            self.save(image_interpolated[:, :, :], 'interpolated.png')
            self.save(image_interpolated[0:int(0.034*self.h), int(0.64*self.w):int(0.67*self.w), :], 'interpolated_zoomed.png')

        plenoptic_image_config = {
            'block_width': 9.927582228653469,
            'block_height': 8.607538408172838
        }

        if self.presentation:
            im = self.baseline_move(image_interpolated, plenoptic_image_config, 0, 0)
            self.save(im, 'baseline.png')

        images = []
        for dx in range(-4, 4, 1):
            # for dy in range(-4, 4, 1):
            image = self.baseline_move(image_interpolated, plenoptic_image_config, dx, dx)
            images.append(image)
        for dx in range(4, -4, -1):
            # for dy in range(-4, 4, 1):
            image = self.baseline_move(image_interpolated, plenoptic_image_config, dx, dx)
            images.append(image)

        src = images[4]

        duration = 0.2
        imageio.mimsave(f'{self.output_path}.gif', images, duration=duration)

        from skimage import exposure
        from skimage.exposure import histogram_matching

        images = []
        for dx in range(-4, 4, 1):
            # for dy in range(-4, 4, 1):
            image = self.baseline_move(image_interpolated, plenoptic_image_config, 0, dx)
            image = exposure.histogram_matching.match_histograms(image, src, multichannel=True)
            images.append(image)
        for dx in range(3, -4, -1):
            # for dy in range(-4, 4, 1):
            image = self.baseline_move(image_interpolated, plenoptic_image_config, 0, dx)
            image = exposure.histogram_matching.match_histograms(image, src, multichannel=True)
            images.append(image)

        duration = 0.1
        imageio.mimsave(f'{self.output_path}_with_histograms.gif', images, duration=duration)

        images3d = []
        for i in range(5):
            image3d = np.zeros_like(images[0])
            image3d[:, :, 0] = images[8 - i][:, :, 0]
            image3d[:, :, 1] = images[i][:, :, 1]
            image3d[:, :, 2] = images[8 - i][:, :, 2]

            images3d.append(image3d)
        for i in range(4, 0, -1):
            image3d = np.zeros_like(images[0])
            image3d[:, :, 0] = images[8 - i][:, :, 0]
            image3d[:, :, 1] = images[i][:, :, 1]
            image3d[:, :, 2] = images[8 - i][:, :, 2]

            images3d.append(image3d)

        self.duration = 0.2
        imageio.mimsave(f'{self.output_path}_3d.gif', images3d, duration=duration)

        if self.presentation:
            plt.figure(figsize=(16, 16))
            plt.imshow(images3d[0])
            plt.savefig(f'{self.output_path}.png')
            self.save(images3d[0], 'frame0.png')

        images3d = []
        for i in range(4):
            image3d = np.zeros_like(images[0])
            image3d[:, :, 0] = images[i + 4][:, :, 0]
            image3d[:, :, 1] = images[i][:, :, 1]
            image3d[:, :, 2] = images[i + 4][:, :, 2]

            images3d.append(image3d)
        for i in range(4, 0, -1):
            image3d = np.zeros_like(images[0])
            image3d[:, :, 0] = images[i + 4][:, :, 0]
            image3d[:, :, 1] = images[i][:, :, 1]
            image3d[:, :, 2] = images[i + 4][:, :, 2]

            images3d.append(image3d)

        self.duration = 0.2
        imageio.mimsave(f'{self.output_path}_3d_view.gif', images3d, duration=self.duration)

    def save(self, im, subscript):
        plt.figure(figsize=(16, 16))
        plt.imshow(im)
        plt.savefig(f'{self.output_path}_{subscript}')

    def get_bayer_pattern_masks(self):
        """
        Возвращает 4 маски заданной высоты и ширины для сырого изображения формата
        B G1
        G2 R
        :return [numpy.ndarray((self.h, self.w)), numpy.ndarray((self.h, self.w)), numpy.ndarray((self.h, self.w)), numpy.ndarray((self.h, self.w))]
        маски для B-компоненты, G1-компоненты, G2-компоненты и R-компоненты
        """
        mask_B = np.array([[1, 0],
                           [0, 0]])

        mask_G1 = np.array([[0, 1],
                            [0, 0]])
        mask_G2 = np.array([[0, 0],
                            [1, 0]])
        mask_R = np.array([[0, 0],
                           [0, 1]])

        mask_B = np.tile(mask_B, (self.h // 2, self.w // 2))
        mask_G1 = np.tile(mask_G1, (self.h // 2, self.w // 2))
        mask_G2 = np.tile(mask_G2, (self.h // 2, self.w // 2))
        mask_R = np.tile(mask_R, (self.h // 2, self.w // 2))

        return mask_B, mask_G1, mask_G2, mask_R

    def high_quality_bayer_interpolation(self, raw_image_in):
        h = raw_image_in.shape[0]
        w = raw_image_in.shape[1]

        weigths_0 = (1 / 8) * np.array([
            [0, 0, 1 / 2, 0, 0],
            [0, -1, 0, -1, 0],
            [-1, 4, 5, 4, -1],
            [0, -1, 0, -1, 0],
            [0, 0, 1 / 2, 0, 0]
        ])

        weigths_1 = (1 / 8) * np.array([
            [0, 0, -1, 0, 0],
            [0, -1, 4, -1, 0],
            [1 / 2, 0, 5, 0, 1 / 2],
            [0, -1, 4, -1, 0],
            [0, 0, -1, 0, 0]
        ])

        weigths_2 = (1 / 8) * np.array([
            [0, 0, -3 / 2, 0, 0],
            [0, 2, 0, 2, 0],
            [-3 / 2, 0, 6, 0, -3 / 2],
            [0, 2, 0, 2, 0],
            [0, 0, -3 / 2, 0, 0]
        ])

        weigths_3 = (1 / 8) * np.array([
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0]
        ])
        # RGB
        image = np.zeros((h, w, 3))
        masks_in = self.get_bayer_pattern_masks()

        # ...
        from copy import copy

        raw_image = copy(raw_image_in)
        masks = copy(masks_in)

        h = raw_image.shape[0]
        w = raw_image.shape[1]

        ext_raw_image = np.zeros((h + 4, w + 4))
        ext_raw_image[2:-2, 2:-2] = raw_image

        # print(masks[0])
        ext_masks = []
        for i in range(4):
            result = np.ones((h + 4, w + 4))
            arr = np.ones((h, w)) - masks[i]
            result[2:-2, 2:-2] -= arr
            ext_masks.append(result)
        ext_masks = {"B": ext_masks[0], "G1": ext_masks[1], "G2": ext_masks[2], "R": ext_masks[3]}

        # print(ext_masks["R"])
        # return
        cur_weigth = None

        for i in range(2, h + 1):
            for j in range(2, w + 1):
                if ext_masks["R"][i, j] > 0:
                    # simple copy
                    image[i - 2, j - 2, 0] = ext_raw_image[i, j]
                else:
                    if ext_masks["R"][i - 1, j - 1] > 0:
                        cur_weigth = weigths_2
                    else:
                        if ext_masks["R"][i - 1, j] > 0:
                            cur_weigth = weigths_1
                        else:
                            cur_weigth = weigths_0

                    pixel = np.sum(ext_raw_image[i - 2:i + 3, j - 2:j + 3] * cur_weigth)

                    image[i - 2, j - 2, 0] = pixel
                    # print("R")

                if ext_masks["G1"][i, j] > 0 or ext_masks["G2"][i, j] > 0:
                    # simple copy
                    image[i - 2, j - 2, 1] = ext_raw_image[i, j]
                else:
                    cur_weigth = weigths_3
                    pixel = np.sum(ext_raw_image[i - 2:i + 3, j - 2:j + 3] * cur_weigth)

                    image[i - 2, j - 2, 1] = pixel
                    # print("G")

                if ext_masks["B"][i, j] > 0:
                    # simple copy
                    image[i - 2, j - 2, 2] = ext_raw_image[i, j]
                else:
                    if ext_masks["B"][i - 1, j - 1] > 0:
                        cur_weigth = weigths_2
                    else:
                        if ext_masks["B"][i - 1, j] > 0:
                            cur_weigth = weigths_1
                        else:
                            cur_weigth = weigths_0

                    pixel = np.sum(ext_raw_image[i - 2:i + 3, j - 2:j + 3] * cur_weigth)

                    image[i - 2, j - 2, 2] = pixel

                    # print("B")
                # print()
        return np.clip(image, 0, 1)

    def baseline_move(self, plenoptic_raw_image, plenoptic_image_config, position_shift_y, position_shift_x):
        raw_image_height = plenoptic_raw_image.shape[0]
        raw_image_width = plenoptic_raw_image.shape[1]
        block_height = plenoptic_image_config['block_height']
        block_width = plenoptic_image_config['block_width']
        block_count_h = int(raw_image_height // block_height)
        block_count_w = int(raw_image_width // block_width)
        result = np.zeros((block_count_h, block_count_w, 3))

        from copy import copy

        def get_block(i, j):
            if i % 2 == 1:
                result = copy(plenoptic_raw_image[int(i * block_height):int((i + 1) * block_height),
                              int(j * block_width):int((j + 1) * block_width)])
            else:
                result = copy(plenoptic_raw_image[int(i * block_height):int((i + 1) * block_height),
                              int(j * block_width + block_width / 2):int((j + 1) * block_width + block_width / 2)])

            return cv2.resize(result,
                              ((int(raw_image_height / block_count_h) + 1) * 2,
                               2 * int(raw_image_width / block_count_w)),
                              interpolation=cv2.INTER_CUBIC)

        # print(raw_image_height/block_count_h, raw_image_width/block_count_w)

        # print(get_block(147, 146).shape)
        for i in range(block_count_h):
            for j in range(block_count_w):
                cur_block = get_block(i, j)
                result[i, j, :] = cur_block[8 + position_shift_y, 8 + position_shift_x, :]

        return np.clip(result, 0, 1)

