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
from AD_classification.config.frequentist_config import BS, AdniDirSkull, DataDir, CV_PathDir, ImageSize, LabelNum, ImageType


class CvImageWriter:
    def __init__(self, AdniDir, DataDir, CV_PathDir, ImageSize, LabelNum, ImageType):
        self.AdniDir = AdniDir
        self.DataDir = DataDir
        self.ImageSize = ImageSize
        self.LabelNum = LabelNum
        self.ImageType = ImageType
        self.CV_PathDir = CV_PathDir

    def write(self):
        if os.path.exists(self.DataDir):                 # Check if any data folder is already available
            def remove_readonly(func, path, excinfo):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(self.DataDir, onerror=remove_readonly)

        if self.LabelNum == 2:                          # CN(0) vs. AD(1)
            pathlib.Path(self.DataDir).mkdir()
            pathlib.Path(self.DataDir + "\\" + str(self.LabelNum) + "-Class").mkdir()
            for i in range(10):
                pathlib.Path(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1)).mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train\\CN").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train\\AD").mkdir()

                trainPaths = np.load(self.CV_PathDir + "\\CN vs AD" + "\\trainX_fold_" + str(i + 1) + ".npy")
                trainLabels = np.load(self.CV_PathDir + "\\CN vs AD" + "\\trainY_fold_" + str(i + 1) + ".npy")
                for (j, (path, label)) in enumerate(zip(trainPaths, trainLabels)):
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
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Train\\CN\\" + path + "-slices" + str(slices[k]) + '.mat', {'im': volume})
                        elif label == 'AD':
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Train\\AD\\" + path + "-slices" + str(slices[k]) + '.mat', {'im': volume})
                        labels.append(label)

                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Validation").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Validation\\CN").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Validation\\AD").mkdir()

                valPaths = np.load(self.CV_PathDir + "\\CN vs AD" + "\\valX_fold_" + str(i + 1) + ".npy")
                valLabels = np.load(self.CV_PathDir + "\\CN vs AD" + "\\valY_fold_" + str(i + 1) + ".npy")
                for (j, (path, label)) in enumerate(zip(valPaths, valLabels)):
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
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Validation\\CN\\" + path + "-slices" + str(slices[k]) + '.mat', {'im': volume})
                        elif label == 'AD':
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Validation\\AD\\" + path + "-slices" + str(slices[k]) + '.mat', {'im': volume})
                        labels.append(label)

        elif self.LabelNum == 3:
            pathlib.Path(self.DataDir).mkdir()
            pathlib.Path(self.DataDir + "\\" + str(self.LabelNum) + "-Class").mkdir()
            for i in range(1):
                pathlib.Path(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1)).mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train\\CN").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train\\MCI").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train\\AD").mkdir()

                trainPaths = np.load(self.CV_PathDir + "\\CN vs MCI vs AD" + "\\trainX_fold_" + str(i + 1) + ".npy")
                trainLabels = np.load(self.CV_PathDir + "\\CN vs MCI vs AD" + "\\trainY_fold_" + str(i + 1) + ".npy")
                for (j, (path, label)) in enumerate(zip(trainPaths, trainLabels)):
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
                            cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Train\\CN\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                        elif label == 'MCI':
                            cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Train\\MCI\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                        else:
                            cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Train\\AD\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                        labels.append(label)

                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Validation").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(
                        i + 1) + "\\Validation\\CN").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(
                        i + 1) + "\\Validation\\MCI").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(
                        i + 1) + "\\Validation\\AD").mkdir()

                valPaths = np.load(self.CV_PathDir + "\\CN vs MCI vs AD" + "\\valX_fold_" + str(i + 1) + ".npy")
                valLabels = np.load(self.CV_PathDir + "\\CN vs MCI vs AD" + "\\valY_fold_" + str(i + 1) + ".npy")
                for (j, (path, label)) in enumerate(zip(valPaths, valLabels)):
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
                            cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Validation\\CN\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                        elif label == 'MCI':
                            cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Validation\\MCI\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                        else:
                            cv2.imwrite(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Validation\\AD\\" + path + "-slices" + str(slices[k]) + '.png', 255.0 * (volume - volume.min()) / (volume.max() - volume.min()))
                        labels.append(label)

        elif self.LabelNum == 5:
            pathlib.Path(self.DataDir).mkdir()
            pathlib.Path(self.DataDir + "\\" + str(self.LabelNum) + "-Class").mkdir()
            for i in range(10):
                pathlib.Path(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1)).mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train\\CN").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train\\SMC").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train\\EMCI").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train\\LMCI").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Train\\AD").mkdir()

                trainPaths = np.load(
                    self.CV_PathDir + "\\CN vs SMC vs EMCI vs LMCI vs AD" + "\\trainX_fold_" + str(i + 1) + ".npy")
                trainLabels = np.load(
                    self.CV_PathDir + "\\CN vs SMC vs EMCI vs LMCI vs AD" + "\\trainY_fold_" + str(i + 1) + ".npy")
                for (j, (path, label)) in enumerate(zip(trainPaths, trainLabels)):
                    volume = os.path.join(self.AdniDir, path).rsplit('.', 1)[0]
                    vol = sio.loadmat(volume)
                    vol = vol.get('im')

                    labels = []
                    for k in range(len(slices)):
                        volume = vol[:, :, slices[k]]
                        if label == 'CN':
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Train\\CN\\" + path + "-slices" + str(slices[k]) + '.mat', {'im': volume})
                        elif label == 'SMC':
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Train\\SMC\\" + path + "-slices" + str(slices[k]) + '.mat', {'im': volume})
                        elif label == 'EMCI':
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Train\\EMCI\\" + path + "-slices" + str(slices[k]) + '.mat', {'im': volume})
                        elif label == 'LMCI':
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Train\\LMCI\\" + path + "-slices" + str(slices[k]) + '.mat', {'im': volume})
                        else:
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Train\\AD\\" + path + "-slices" + str(slices[k]) + '.mat', {'im': volume})
                        labels.append(label)

                np.save(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\train_labels", label)

                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) + "\\Validation").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(
                        i + 1) + "\\Validation\\CN").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(
                        i + 1) + "\\Validation\\SMC").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(
                        i + 1) + "\\Validation\\EMCI").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(
                        i + 1) + "\\Validation\\LMCI").mkdir()
                pathlib.Path(
                    self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(
                        i + 1) + "\\Validation\\AD").mkdir()

                valPaths = np.load(
                    self.CV_PathDir + "\\CN vs SMC vs EMCI vs LMCI vs AD" + "\\valX_fold_" + str(i + 1) + ".npy")
                valLabels = np.load(
                    self.CV_PathDir + "\\CN vs SMC vs EMCI vs LMCI vs AD" + "\\valY_fold_" + str(i + 1) + ".npy")
                for (j, (path, label)) in enumerate(zip(valPaths, valLabels)):
                    volume = os.path.join(self.AdniDir, path).rsplit('.', 1)[0]
                    vol = sio.loadmat(volume)
                    vol = vol.get('im')

                    labels = []
                    for k in range(len(slices)):
                        volume = vol[:, :, slices[k]]
                        if valLabels == 'CN':
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Validation\\CN\\" + path + "-slices" + str(slices[k]) + '.mat', {'im': volume})
                        elif label == 'SMC':
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Validation\\SMC\\" + path + "-slices" + str(slices[k]) + '.mat',
                                        {'im': volume})
                        elif label == 'EMCI':
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Validation\\EMCI\\" + path + "-slices" + str(slices[k]) + '.mat',
                                        {'im': volume})
                        elif label == 'LMCI':
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Validation\\LMCI\\" + path + "-slices" + str(slices[k]) + '.mat',
                                        {'im': volume})
                        else:
                            sio.savemat(self.DataDir + "\\" + str(self.LabelNum) + "-Class" + "\\Fold_" + str(i + 1) +
                                        "\\Validation\\AD\\" + path + "-slices" + str(slices[k]) + '.mat', {'im': volume})
                        labels.append(label)
                np.save(
                    self.DataDir + "\\" + self.LabelNum + "-Class" + "\\Fold_" + str(i + 1) + "\\val_labels", label)


w = CvImageWriter(AdniDirSkull, DataDir, CV_PathDir, ImageSize, LabelNum, ImageType)
w.write()