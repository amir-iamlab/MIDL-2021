# import the necessary packages

import numpy as np
import nibabel as nib
from AD_classification.config import frequentist_config


class DbIdxFind:
    def __init__(self, Paths):
        self.paths = Paths

    def find(self):

        # initialize the counter count
        counter = 0
        for path in np.arange(self.paths):
            # load the image and process it
            image = nib.load(path)
            image = image.get_fdata()
            image = image[:, :, frequentist_config.slices[0]:frequentist_config.slices[1]]
            for j in range(image.shape[2]):
                counter = counter + 1

        yield counter
