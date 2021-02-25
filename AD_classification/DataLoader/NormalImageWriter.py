import os
import pathlib
from glob import glob
import pandas as pd
import numpy as np
import scipy.io as sio
import stat
import shutil
import cv2
import imageio
from AD_classification.config.frequentist_config import slices
from PIL import Image
from AD_classification.preprocessor.centercrop import centercrop, crop_generator
from AD_classification.preprocessor.aspectawareresize import aspectawareresize
from AD_classification.config.frequentist_config import AdniDirSkull,AdniDirSkull_new, DataDir,DataDir_new, Normal_PathDir, ImageSize, LabelNum, ImageType, TrainDir, slice_num, CCNADirSkull


class NormalImageWriter:
    def __init__(self, AdniDir, DataDir, Normal_PathDir, ImageSize, LabelNum, ImageType):
        self.AdniDir = AdniDir
        self.DataDir = DataDir
        self.ImageSize = ImageSize
        self.LabelNum = LabelNum
        self.ImageType = ImageType
        self.Normal_PathDir = Normal_PathDir

    def write(self):
        aaresize = aspectawareresize(ImageSize, ImageSize)
        # if os.path.exists(self.DataDir):                 # Check if any data folder is already available
        #     def remove_readonly(func, path, excinfo):
        #         os.chmod(path, stat.S_IWRITE)
        #         func(path)
        #     shutil.rmtree(self.DataDir, onerror=remove_readonly)

        # pathlib.Path(self.DataDir).mkdir()
        if self.LabelNum == 2:                          # CN(0) vs. AD(1)

            pathlib.Path(self.DataDir + "\\" + str(self.LabelNum) + "-Class").mkdir()
            pathlib.Path(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num).mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train\\CN").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train\\AD").mkdir()

            # trainPaths = np.load(self.Normal_PathDir + "\\CN vs AD" + "\\trainX" + ".npy")
            # trainLabels = np.load(self.Normal_PathDir + "\\CN vs AD" + "\\trainY" + ".npy")
            # for (j, (path, label)) in enumerate(zip(trainPaths, trainLabels)):
            #     volume = os.path.join(self.AdniDir, path).rsplit('.', 1)[0]
            #     vol = sio.loadmat(volume)
            #     vol = vol.get('im')
            #     # if self.ImageType == 0:
            #     #     vol = sio.loadmat(volume)
            #     #     vol = vol.get('im')[0][0][0]
            #     # elif self.ImageType == 1:
            #     #     vol = sio.loadmat(volume)
            #     #     vol = vol.get('im')[0][0][1]
            #     # else:
            #     #     vol = sio.loadmat(volume)
            #     #     vol = vol.get('regvol')
            #
            #     labels = []
            #     for k in range(len(slices)):
            #         volume = vol[:, :, slices[k]]
            #         if label == 'CN':
            #             cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
            #                         "\\Train\\CN\\" + path + "-slices" + str(slices[k]) + '.png',
            #                         255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
            #         else:
            #             cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
            #                         "\\Train\\AD\\" + path + "-slices" + str(slices[k]) + '.png',
            #                         255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
            #         labels.append(label)
            #
            # pathlib.Path(
            #     self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation").mkdir()
            # pathlib.Path(
            #     self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation\\CN").mkdir()
            # pathlib.Path(
            #     self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation\\AD").mkdir()
            #
            # valPaths = np.load(self.Normal_PathDir + "\\CN vs AD" + "\\validationX" + ".npy")
            # valLabels = np.load(self.Normal_PathDir + "\\CN vs AD" + "\\validationY" + ".npy")
            # for (j, (path, label)) in enumerate(zip(valPaths, valLabels)):
            #     volume = os.path.join(self.AdniDir, path).rsplit('.', 1)[0]
            #     vol = sio.loadmat(volume)
            #     vol = vol.get('im')
            #     # if self.ImageType == 0:
            #     #     vol = sio.loadmat(volume)
            #     #     vol = vol.get('im')[0][0][0]
            #     # elif self.ImageType == 1:
            #     #     vol = sio.loadmat(volume)
            #     #     vol = vol.get('im')[0][0][1]
            #     # else:
            #     #     vol = sio.loadmat(volume)
            #     #     vol = vol.get('regvol')
            #
            #     labels = []
            #     for k in range(len(slices)):
            #         volume = vol[:, :, slices[k]]
            #         if label == 'CN':
            #             cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
            #                         "\\Validation\\CN\\" + path + "-slices" + str(slices[k]) + '.png',
            #                         255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
            #         else:
            #             cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
            #                         "\\Validation\\AD\\" + path + "-slices" + str(slices[k]) + '.png',
            #                         255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
            #         labels.append(label)

            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" +slice_num +  "\\Test").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" +slice_num +  "\\Test\\CN").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" +slice_num +  "\\Test\\AD").mkdir()

            testPaths = np.load(self.Normal_PathDir + "\\CN vs AD" + "\\testX" + ".npy")
            testLabels = np.load(self.Normal_PathDir + "\\CN vs AD" + "\\testY" + ".npy")
            for (j, (path, label)) in enumerate(zip(testPaths, testLabels)):
                volume = os.path.join(self.AdniDir, path).rsplit('.', 1)[0]
                vol = sio.loadmat(volume)
                vol = vol.get('im')
                # if self.ImageType == 0:
                #     vol = sio.loadmat(volume)
                #     vol = vol.get('im')[0][0][0]
                # elif self.ImageType == 1:
                #     vol = sio.loadmat(volume)
                #     vol = vol.get('im')[0][0][1]
                # else:
                #     vol = sio.loadmat(volume)
                #     vol = vol.get('regvol')

                labels = []
                for k in range(len(slices)):
                    volume = vol[:, :, slices[k]]
                    if label == 'CN':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
                                    "\\Test\\CN\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    else:
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
                                    "\\Test\\AD\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    labels.append(label)

        elif self.LabelNum == 3:
            # pathlib.Path(self.DataDir).mkdir()
            # pathlib.Path(self.DataDir + "\\" + str(self.LabelNum) + "-Class").mkdir()
            pathlib.Path(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num).mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train\\CN").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train\\MCI").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train\\AD").mkdir()

            trainPaths = np.load(self.Normal_PathDir + "\\CN vs MCI vs AD" + "\\trainX" + ".npy")
            trainLabels = np.load(self.Normal_PathDir + "\\CN vs MCI vs AD" + "\\trainY" + ".npy")
            for (j, (path, label)) in enumerate(zip(trainPaths, trainLabels)):
                volume = os.path.join(self.AdniDir, path).rsplit('.', 1)[0]
                vol = sio.loadmat(volume)
                vol = vol.get('im')
                # vol = aaresize.preprocess(vol)
                # vol = centercrop(vol)
                # if self.ImageType == 0:
                #     vol = sio.loadmat(volume)
                #     vol = vol.get('im')[0][0][0]
                # elif self.ImageType == 1:
                #     vol = sio.loadmat(volume)
                #     vol = vol.get('im')[0][0][1]
                # else:
                #     vol = sio.loadmat(volume)
                #     vol = vol.get('regvol')

                labels = []
                for k in range(len(slices)):
                    volume = vol[:, :, slices[k]]
                    if label == 'CN':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
                                    "\\Train\\CN\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    elif label == 'MCI':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
                                    "\\Train\\MCI\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    else:
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
                                    "\\Train\\AD\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    labels.append(label)

            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation\\CN").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation\\MCI").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation\\AD").mkdir()

            valPaths = np.load(self.Normal_PathDir + "\\CN vs MCI vs AD" + "\\validationX" + ".npy")
            valLabels = np.load(self.Normal_PathDir + "\\CN vs MCI vs AD" + "\\validationY" + ".npy")
            for (j, (path, label)) in enumerate(zip(valPaths, valLabels)):
                volume = os.path.join(self.AdniDir, path).rsplit('.', 1)[0]
                vol = sio.loadmat(volume)
                vol = vol.get('im')
                # vol = aaresize.preprocess(vol)
                # vol = centercrop(vol)
                # if self.ImageType == 0:
                #     vol = sio.loadmat(volume)
                #     vol = vol.get('im')[0][0][0]
                # elif self.ImageType == 1:
                #     vol = sio.loadmat(volume)
                #     vol = vol.get('im')[0][0][1]
                # else:
                #     vol = sio.loadmat(volume)
                #     vol = vol.get('regvol')

                labels = []
                for k in range(len(slices)):
                    volume = vol[:, :, slices[k]]
                    if label == 'CN':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
                                    "\\Validation\\CN\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    elif label == 'MCI':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
                                    "\\Validation\\MCI\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    else:
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
                                    "\\Validation\\AD\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    labels.append(label)

            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Test").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Test\\CN").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Test\\MCI").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Test\\AD").mkdir()

            testPaths = np.load(self.Normal_PathDir + "\\CN vs MCI vs AD" + "\\testX" + ".npy")
            testLabels = np.load(self.Normal_PathDir + "\\CN vs MCI vs AD" + "\\testY" + ".npy")
            for (j, (path, label)) in enumerate(zip(testPaths, testLabels)):
                volume = os.path.join(self.AdniDir, path).rsplit('.', 1)[0]
                vol = sio.loadmat(volume)
                vol = vol.get('im')
                # vol = aaresize.preprocess(vol)
                # vol = centercrop(vol)
                # if self.ImageType == 0:
                #     vol = sio.loadmat(volume)
                #     vol = vol.get('im')[0][0][0]
                # elif self.ImageType == 1:
                #     vol = sio.loadmat(volume)
                #     vol = vol.get('im')[0][0][1]
                # else:
                #     vol = sio.loadmat(volume)
                #     vol = vol.get('regvol')

                labels = []
                for k in range(len(slices)):
                    volume = vol[:, :, slices[k]]
                    if label == 'CN':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
                                    "\\Test\\CN\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    elif label == 'MCI':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
                                    "\\Test\\MCI\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    else:
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num +
                                    "\\Test\\AD\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    labels.append(label)

        elif self.LabelNum == 5:
            # pathlib.Path(self.DataDir).mkdir()
            pathlib.Path(self.DataDir + "\\" + str(self.LabelNum) + "-Class").mkdir()
            pathlib.Path(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num).mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train\\CN").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train\\SMC").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train\\EMCI").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train\\LMCI").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Train\\AD").mkdir()

            trainPaths = np.load(
                self.Normal_PathDir + "\\CN vs SMC vs EMCI vs LMCI vs AD" + "\\trainX" + ".npy")
            trainLabels = np.load(
                self.Normal_PathDir + "\\CN vs SMC vs EMCI vs LMCI vs AD" + "\\trainY" + ".npy")
            for (j, (path, label)) in enumerate(zip(trainPaths, trainLabels)):
                volume = os.path.join(self.AdniDir, path).rsplit('.', 1)[0]
                vol = sio.loadmat(volume)
                vol = vol.get('im')

                # vol = centercrop(vol)

                labels = []
                for k in range(len(slices)):
                    volume = vol[:, :, slices[k]]
                    if label == 'CN':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Train\\CN\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    elif label == 'SMC':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Train\\SMC\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    elif label == 'EMCI':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Train\\EMCI\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    elif label == 'LMCI':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Train\\LMCI\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    else:
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Train\\AD\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    labels.append(label)

            # np.save(
            #     self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\train_labels", label)

            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation\\CN").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation\\SMC").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation\\EMCI").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation\\LMCI").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Validation\\AD").mkdir()

            valPaths = np.load(
                self.Normal_PathDir + "\\CN vs SMC vs EMCI vs LMCI vs AD" + "\\validationX" + ".npy")
            valLabels = np.load(
                self.Normal_PathDir + "\\CN vs SMC vs EMCI vs LMCI vs AD" + "\\validationY" + ".npy")
            for (j, (path, label)) in enumerate(zip(valPaths, valLabels)):
                volume = os.path.join(self.AdniDir, path).rsplit('.', 1)[0]
                vol = sio.loadmat(volume)
                vol = vol.get('im')
                # vol = centercrop(vol)

                labels = []
                for k in range(len(slices)):
                    volume = vol[:, :, slices[k]]
                    if label == 'CN':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Validation\\CN\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    elif label == 'SMC':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Validation\\SMC\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    elif label == 'EMCI':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Validation\\EMCI\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    elif label == 'LMCI':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Validation\\LMCI\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    else:
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Validation\\AD\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    labels.append(label)

            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Test").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Test\\CN").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Test\\SMC").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Test\\EMCI").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Test\\LMCI").mkdir()
            pathlib.Path(
                self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + "\\Test\\AD").mkdir()

            testPaths = np.load(
                self.Normal_PathDir + "\\CN vs SMC vs EMCI vs LMCI vs AD" + "\\testX" + ".npy")
            testLabels = np.load(
                self.Normal_PathDir + "\\CN vs SMC vs EMCI vs LMCI vs AD" + "\\testY" + ".npy")
            for (j, (path, label)) in enumerate(zip(testPaths, testLabels)):
                volume = os.path.join(self.AdniDir, path).rsplit('.', 1)[0]
                vol = sio.loadmat(volume)
                vol = vol.get('im')
                # vol = centercrop(vol)

                labels = []
                for k in range(len(slices)):
                    volume = vol[:, :, slices[k]]
                    if label == 'CN':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Test\\CN\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    elif label == 'SMC':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Test\\SMC\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    elif label == 'EMCI':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Test\\EMCI\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    elif label == 'LMCI':
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Test\\LMCI\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    else:
                        cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + slice_num + 
                                    "\\Test\\AD\\" + path + "-slices" + str(slices[k]) + '.png',
                                    255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                    labels.append(label)
            # np.save(
            #     self.DataDir + "\\" + self.LabelNum + "-Class" + "\\Fold_" + str(i + 1) + "\\val_labels", label)

    # def oversample(self):
    #
    #     if self.LabelNum == 3:
    #         filepaths = []
    #         cn_data = glob(TrainDir + '\\CN' + '\\*')
    #         for file in range(len(cn_data)):
    #             new = os.listdir(TrainDir)[file].rsplit('.', 1)[0]

# #
# DataDir_new='C:\\RyeU\\PhD\\Thesis\\Keras\\CCNA'
# Normal_PathDir = 'C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\CCNAPath'
#
# w = NormalImageWriter(CCNADirSkull, DataDir_new, Normal_PathDir, ImageSize, LabelNum, ImageType)
# w.write()