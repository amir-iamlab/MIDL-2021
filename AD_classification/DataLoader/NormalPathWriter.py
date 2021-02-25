# import the necessary packages
from AD_classification.config.frequentist_config import Normal_PathDir, LabelNum, LabelDir, AdniDir
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
from glob import glob
import pandas as pd
import stat
import pathlib
import os
import shutil
from sklearn.model_selection import train_test_split

# if os.path.exists(CV_PathDir):  # Check if any data folder is already available
#     def remove_readonly(func, path, excinfo):
#         os.chmod(path, stat.S_IWRITE)
#         func(path)
#
#
#     shutil.rmtree(CV_PathDir, onerror=remove_readonly)
Normal_PathDir = 'C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\CCNAPath'
LabelDir = 'C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\Data\\CCNA_Clinical_Data.xlsx'
AdniDir =  'C:\\RyeU\\PhD\\Thesis\\Dataset\\ADNI\\Data\\CCNA'
if LabelNum == 2:
    pathlib.Path(Normal_PathDir + "\\CN vs AD").mkdir()
elif LabelNum == 3:
    pathlib.Path(Normal_PathDir + "\\CN vs MCI vs AD").mkdir()
else:
    pathlib.Path(Normal_PathDir + "\\CN vs SMC vs EMCI vs LMCI vs AD").mkdir()

# name = glob(CV_PathDir + '\\*')
# for f in name:
#     os.remove(f)

if LabelNum == 2:

    df = pd.read_excel(LabelDir)
    for i in range(len(df)):
        if df.Diagnosis[i] == 'SCI':
            df.Diagnosis[i] = 'CN'
    new_df = df.loc[df['Diagnosis'].isin(['CN', 'AD'])]
    new_df = new_df.to_numpy()
    labels = []
    filepaths = []
    data = glob(AdniDir + '\\*')
    for file in range(len(data)):
        new = os.listdir(AdniDir)[file].rsplit('.', 1)[0]
        for j in range(len(new_df)):
            if new == new_df[j, 0]:
            # if new == new_df[j, 2]:
                filepaths.append(new)
                break

    for i in range(len(filepaths)):
        for j in range(len(new_df)):
            if new_df[j, 0] == filepaths[i].rsplit('.', 1)[0]:
                #new_df[j, 2] == filepaths[i].rsplit('.', 1)[0]:
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

    # split = train_test_split(filepaths, filelabels, test_size=.2, stratify=filelabels, random_state=42)
    # (trainPaths, valPaths, trainLabels, valLabels) = split
    # split2 = train_test_split(valPaths, valLabels, test_size=.5, stratify=valLabels, random_state=42)
    # (testPaths, valPaths, testLabels, valLabels) = split2
    testPaths = filepaths
    testLabels = filelabels
    # print('n train samples', len(trainPaths))
    # print('n valid samples', len(valPaths))
    print('n test samples', len(testPaths))
    # print('n train labels', len(trainLabels))
    # print('n valid labels', len(valLabels))
    print('n test labels', len(testLabels))
    # np.save(Normal_PathDir + '\\CN vs AD\\trainX' + '.npy', trainPaths)
    # np.save(Normal_PathDir + '\\CN vs AD\\trainY' + '.npy', trainLabels)
    # np.save(Normal_PathDir + '\\CN vs AD\\validationX' + '.npy', valPaths)
    # np.save(Normal_PathDir + '\\CN vs AD\\validationY' + '.npy', valLabels)
    np.save(Normal_PathDir + '\\CN vs AD\\testX' + '.npy', testPaths)
    np.save(Normal_PathDir + '\\CN vs AD\\testY' + '.npy', testLabels)

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
    split = train_test_split(filepaths, filelabels, test_size=.2, stratify=filelabels, random_state=42)
    (trainPaths, valPaths, trainLabels, valLabels) = split
    split2 = train_test_split(valPaths, valLabels, test_size=.5, stratify=valLabels, random_state=42)
    (testPaths, valPaths, testLabels, valLabels) = split2
    print('n train samples', len(trainPaths))
    print('n valid samples', len(valPaths))
    print('n test samples', len(testPaths))
    print('n train labels', len(trainLabels))
    print('n valid labels', len(valLabels))
    print('n test labels', len(testLabels))

    np.save(Normal_PathDir + '\\CN vs MCI vs AD\\trainX' + '.npy', trainPaths)
    np.save(Normal_PathDir + '\\CN vs MCI vs AD\\trainY' + '.npy', trainLabels)
    np.save(Normal_PathDir + '\\CN vs MCI vs AD\\validationX' + '.npy', valPaths)
    np.save(Normal_PathDir + '\\CN vs MCI vs AD\\validationY' + '.npy', valLabels)
    np.save(Normal_PathDir + '\\CN vs MCI vs AD\\testX' + '.npy', testPaths)
    np.save(Normal_PathDir + '\\CN vs MCI vs AD\\testY' + '.npy', testLabels)

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
    split = train_test_split(filepaths, filelabels, test_size=.2, stratify=filelabels, random_state=42)
    (trainPaths, valPaths, trainLabels, valLabels) = split
    split2 = train_test_split(valPaths, valLabels, test_size=.5, stratify=valLabels, random_state=42)
    (testPaths, valPaths, testLabels, valLabels) = split2
    print('n train samples', len(trainPaths))
    print('n valid samples', len(valPaths))
    print('n test samples', len(testPaths))
    print('n train labels', len(trainLabels))
    print('n valid labels', len(valLabels))
    print('n test labels', len(testLabels))
    np.save(Normal_PathDir + '\\CN vs SMC vs EMCI vs LMCI vs AD\\trainX' + '.npy', trainPaths)
    np.save(Normal_PathDir + '\\CN vs SMC vs EMCI vs LMCI vs AD\\trainY' + '.npy', trainLabels)
    np.save(Normal_PathDir + '\\CN vs SMC vs EMCI vs LMCI vs AD\\validationX' + '.npy', valPaths)
    np.save(Normal_PathDir + '\\CN vs SMC vs EMCI vs LMCI vs AD\\validationY' + '.npy', valLabels)
    np.save(Normal_PathDir + '\\CN vs SMC vs EMCI vs LMCI vs AD\\testX' + '.npy', testPaths)
    np.save(Normal_PathDir + '\\CN vs SMC vs EMCI vs LMCI vs AD\\testY' + '.npy', testLabels)