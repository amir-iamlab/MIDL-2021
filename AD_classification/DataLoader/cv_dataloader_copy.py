from keras.utils import np_utils
import numpy as np
import nibabel as nib
from AD_classification.config import frequentist_config
import scipy.io as sio


# class CV_DataLoader:
#     def __init__(self, xpath, ypath, batchsize, preprocessors=None, aug=None, binarize=True, classes=3):
#         xpath = np.load(xpath)
#         ypath = np.load(ypath)
#         self.xpath = xpath
#         self.ypath = ypath
#         self.batchsize = batchsize
#         self.preprocessors = preprocessors
#         self.aug = aug
#         self.binarize = binarize
#         self.classes = classes
#         # open the HDF5 database for reading and determine the total  number of entries in the database
#         self.numImages = self.xpath.shape[0]

def data_generator(xpath, ypath, batchsize, preprocessors=None, aug=None, binarize=True, classes=3):
    xpath = np.load(xpath)
    ypath = np.load(ypath)
    # self.xpath = xpath
    # self.ypath = ypath
    # self.batchsize = batchsize
    # self.preprocessors = preprocessors
    # self.aug = aug
    # self.binarize = binarize
    # self.classes = classes
    # open the HDF5 database for reading and determine the total  number of entries in the database
    numImages = xpath.shape[0]
    if xpath[0].rsplit('.', 1)[1] == 'nii':
        for start in np.arange(0, numImages, batchsize):
            # extract the images and labels from the path for .nii version
            xbatch = []
            ybatch = []
            end = min(start + batchsize, numImages)
            #ids_batch = self.xpath[i:end]
            for i in range(start, end):
                # load the image and process it
                image = nib.load(xpath[i])
                image = image.get_fdata()
                label = ypath[i]
                if binarize:
                    label = np_utils.to_categorical(label, classes)
                for j in range(len(frequentist_config.slices)):
                    img = image[:, :, frequentist_config.slices[j]]

                    if preprocessors is not None:
                        # loop over the preprocessors and apply each to the image
                        for p in preprocessors:
                            img = p.preprocess(img)
                            img.shape
                            #img = np.reshape(img, (AD_config.SIZE, AD_config.SIZE, 1))
                            img.shape
                        # update the images array to be the processed images
                        img = np.array(img)
                        img = np.stack((img,) * 3, axis=-1)
                    # if the data augmenator exists, apply it
                    if aug is not None:
                        (img, label) = next(aug.flow(img, label, batch_size=batchsize))

                    xbatch.append(img)
                    ybatch.append(label)
            # yield images and labels
            xbatch = np.array(xbatch)
            ybatch = np.array(ybatch)
            yield xbatch, ybatch
    else:
        for start in np.arange(0, numImages, batchsize):

            # extract the images and labels from the path fot .mat version
            xbatch = []
            ybatch = []
            end = min(start + batchsize, numImages)
            # ids_batch = self.xpath[i:end]
            for i in range(start, end):
                # load the image and process it
                image = sio.loadmat(xpath[i])
                image = np.array(image.get('regvol'))
                label = ypath[i]
                if binarize:
                    label = np_utils.to_categorical(label, classes)
                for j in range(len(frequentist_config.slices)):
                    img = image[:, :, frequentist_config.slices[j]]
                    #
                    if preprocessors is not None:
                        # loop over the preprocessors and apply each to the image
                        for p in preprocessors:
                            img = p.preprocess(img)
                            img = np.reshape(img, (frequentist_config.SIZE, frequentist_config.SIZE, 1))
                        # update the images array to be the processed images
                        img = np.array(img)

                    # if the data augmenator exists, apply it
                    if aug is not None:
                        (img, label) = next(aug.flow(img, label, batch_size=batchsize))

                    xbatch.append(img)
                    ybatch.append(label)
            # yield images and labels
            xbatch = np.array(xbatch)
            ybatch = np.array(ybatch)
            yield xbatch, ybatch
