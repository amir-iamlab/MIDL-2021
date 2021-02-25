# import the necessary packages
from AD_classification.config.frequentist_config import CV_PathDir, LabelNum, LabelDir, AdniDir, k_fold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
from glob import glob
import pandas as pd
import stat
import pathlib
import os
import shutil
from sklearn.model_selection import StratifiedKFold

# if os.path.exists(CV_PathDir):  # Check if any data folder is already available
#     def remove_readonly(func, path, excinfo):
#         os.chmod(path, stat.S_IWRITE)
#         func(path)
#
#
#     shutil.rmtree(CV_PathDir, onerror=remove_readonly)

if LabelNum == 2:
    pathlib.Path(CV_PathDir + "\\CN vs AD").mkdir()
elif LabelNum == 3:
    pathlib.Path(CV_PathDir + "\\CN vs MCI vs AD").mkdir()
else:
    pathlib.Path(CV_PathDir + "\\CN vs SMC vs EMCI vs LMCI vs AD").mkdir()

# name = glob(CV_PathDir + '\\*')
# for f in name:
#     os.remove(f)

if LabelNum == 2:

    df = pd.read_csv(LabelDir)
    for i in range(len(df)):
        if df.Diagnosis[i] == 'SMC':
            df.Diagnosis[i] = 'CN'
    new_df = df.loc[df['Diagnosis'].isin(['CN', 'AD'])]
    new_df = new_df.to_numpy()
    labels = []
    filepaths = []
    data = glob(AdniDir + '\\*')
    for file in range(len(data)):
        new = os.listdir(AdniDir)[file].rsplit('.', 1)[0]
        for j in range(len(new_df)):
            if new == new_df[j, 2]:
                filepaths.append(new)
                break

    for i in range(len(filepaths)):
        for j in range(len(new_df)):
            if new_df[j, 2] == filepaths[i].rsplit('.', 1)[0]:
                labels.append(new_df[j, 5])
                break
    filelabels = np.array(labels).astype("str")

    # perform stratified sampling from the training set to build the testing split from the training data
    random.seed(42)
    combined = list(zip(filepaths, filelabels))
    random.shuffle(combined)
    filepaths, filelabels = zip(*combined)
    filepaths = np.array(filepaths)
    filelabels = np.array(filelabels)
    foldnum = k_fold
    skf = StratifiedKFold(n_splits=foldnum, shuffle=False)
    for index, (train_indices, val_indices) in enumerate(skf.split(filepaths, filelabels)):
        if index > -1:
            index = index + 1
            print("Building fold " + str(index) + "/10...")
            trainPaths, valPaths = filepaths[train_indices], filepaths[val_indices]
            trainLabels, valLabels = filelabels[train_indices], filelabels[val_indices]
            print('n train samples', len(trainPaths))
            print('n valid samples', len(valPaths))
            print('n train labels', len(trainLabels))
            print('n valid labels', len(valLabels))
            np.save(CV_PathDir + '\\CN vs AD\\trainX_fold' + "_" + str(index) + '.npy', trainPaths)
            np.save(CV_PathDir + '\\CN vs AD\\trainY_fold' + "_" + str(index) + '.npy', trainLabels)
            np.save(CV_PathDir + '\\CN vs AD\\valX_fold' + "_" + str(index) + '.npy', valPaths)
            np.save(CV_PathDir + '\\CN vs AD\\valY_fold' + "_" + str(index) + '.npy', valLabels)

elif LabelNum == 3:

    df = pd.read_csv(LabelDir)
    for i in range(len(df)):
        if df.Diagnosis[i] == 'EMCI':
            df.Diagnosis[i] = 'MCI'
        if df.Diagnosis[i] == 'LMCI':
            df.Diagnosis[i] = 'MCI'
        if df.Diagnosis[i] == 'SMC':
            df.Diagnosis[i] = 'CN'
    new_df = df.loc[df['Diagnosis'].isin(['CN', 'MCI', 'AD'])]
    new_df = new_df.to_numpy()
    labels = []
    filepaths = []
    data = glob(AdniDir + '\\*')
    for file in range(len(data)):
        new = os.listdir(AdniDir)[file].rsplit('.', 1)[0]
        for j in range(len(new_df)):
            if new == new_df[j, 2]:
                filepaths.append(new)
                break

    for i in range(len(filepaths)):
        for j in range(len(new_df)):
            if new_df[j, 2] == filepaths[i].rsplit('.', 1)[0]:
                labels.append(new_df[j, 5])
                break
    filelabels = np.array(labels).astype("str")

    # perform stratified sampling from the training set to build the testing split from the training data
    random.seed(42)
    combined = list(zip(filepaths, filelabels))
    random.shuffle(combined)
    filepaths, filelabels = zip(*combined)
    filepaths = np.array(filepaths)
    filelabels = np.array(filelabels)
    foldnum = k_fold
    skf = StratifiedKFold(n_splits=foldnum, shuffle=False)
    for index, (train_indices, val_indices) in enumerate(skf.split(filepaths, filelabels)):
        if index > -1:
            index = index + 1
            print("Building fold " + str(index) + "/10...")
            trainPaths, valPaths = filepaths[train_indices], filepaths[val_indices]
            trainLabels, valLabels = filelabels[train_indices], filelabels[val_indices]
            print('n train samples', len(trainPaths))
            print('n valid samples', len(valPaths))
            print('n train labels', len(trainLabels))
            print('n valid labels', len(valLabels))
            np.save(CV_PathDir + '\\CN vs MCI vs AD\\trainX_fold' + "_" + str(index) + '.npy', trainPaths)
            np.save(CV_PathDir + '\\CN vs MCI vs AD\\trainY_fold' + "_" + str(index) + '.npy', trainLabels)
            np.save(CV_PathDir + '\\CN vs MCI vs AD\\valX_fold' + "_" + str(index) + '.npy', valPaths)
            np.save(CV_PathDir + '\\CN vs MCI vs AD\\valY_fold' + "_" + str(index) + '.npy', valLabels)
else:
    df = pd.read_csv(LabelDir)
    new_df = df.to_numpy()
    labels = []
    filepaths = []
    data = glob(AdniDir + '\\*')
    for file in range(len(data)):
        new = os.listdir(AdniDir)[file].rsplit('.', 1)[0]
        for j in range(len(new_df)):
            if new == new_df[j, 2]:
                filepaths.append(new)
                break

    for i in range(len(filepaths)):
        for j in range(len(new_df)):
            if new_df[j, 2] == filepaths[i].rsplit('.', 1)[0]:
                labels.append(new_df[j, 5])
                break
    filelabels = np.array(labels).astype("str")

    # perform stratified sampling from the training set to build the testing split from the training data
    random.seed(42)
    combined = list(zip(filepaths, filelabels))
    random.shuffle(combined)
    filepaths, filelabels = zip(*combined)
    filepaths = np.array(filepaths)
    filelabels = np.array(filelabels)
    foldnum = k_fold
    skf = StratifiedKFold(n_splits=foldnum, shuffle=False)
    for index, (train_indices, val_indices) in enumerate(skf.split(filepaths, filelabels)):
        if index > -1:
            index = index + 1
            print("Building fold " + str(index) + "/10...")
            trainPaths, valPaths = filepaths[train_indices], filepaths[val_indices]
            trainLabels, valLabels = filelabels[train_indices], filelabels[val_indices]
            print('n train samples', len(trainPaths))
            print('n valid samples', len(valPaths))
            print('n train labels', len(trainLabels))
            print('n valid labels', len(valLabels))
            np.save(CV_PathDir + '\\CN vs SMC vs EMCI vs LMCI vs AD\\trainX_fold' + "_" + str(index) + '.npy', trainPaths)
            np.save(CV_PathDir + '\\CN vs SMC vs EMCI vs LMCI vs AD\\trainY_fold' + "_" + str(index) + '.npy', trainLabels)
            np.save(CV_PathDir + '\\CN vs SMC vs EMCI vs LMCI vs AD\\valX_fold' + "_" + str(index) + '.npy', valPaths)
            np.save(CV_PathDir + '\\CN vs SMC vs EMCI vs LMCI vs AD\\valY_fold' + "_" + str(index) + '.npy', valLabels)