from PIL import Image
import numpy as np


def centercrop(image, shape=(224, 224)):
    # width, height = image.size  # Get dimensions
    #
    # left = (width - shape) / 2
    # top = (height - shape) / 2
    # right = (width + shape) / 2
    # bottom = (height + shape) / 2
    #
    # # Crop the center of the image
    # im = image.crop((left, top, right, bottom))
    # return im
    centerw, centerh = image.shape[0] // 2, image.shape[1] // 2
    halfw, halfh = shape[0] // 2, shape[1] // 2
    return image[centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh]


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = centercrop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)