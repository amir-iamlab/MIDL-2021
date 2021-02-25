import numpy as np
import cv2
import os
import glob
import pathlib
from PIL import Image
TrainDir = "C:\\RyeU\\PhD\\Thesis\\Keras\\Dataset\\3-Class\\Train"
CNRes = "C:\\RyeU\\PhD\\Thesis\\Keras\\Dataset\\3-Class\\Train\\Oversampled\\CN"
ADRes = "C:\\RyeU\\PhD\\Thesis\\Keras\\Dataset\\3-Class\\Train\\Oversampled\\AD"

pathlib.Path("C:\\RyeU\\PhD\\Thesis\\Keras\\Dataset\\3-Class\\Train\\Oversampled").mkdir()
pathlib.Path(CNRes).mkdir()
pathlib.Path(ADRes).mkdir()

cn_filepaths = []
cn_data = glob.glob(TrainDir + '\\CN' + '\\*')
for file in np.arange(len(cn_data)):
    new = os.listdir(TrainDir + '\\CN\\')[file].rsplit('.', 1)[0]
    cn_filepaths.append(new)

mci_data = glob.glob(TrainDir + '\\MCI' + '\\*')

ad_filepaths = []
ad_data = glob.glob(TrainDir + '\\AD' + '\\*')
for file in np.arange(len(ad_data)):
    new = os.listdir(TrainDir + '\\AD\\')[file].rsplit('.', 1)[0]
    ad_filepaths.append(new)
print('cn images: ', len(cn_data))
print('mci images: ', len(mci_data))
print('ad images: ', len(ad_data))
cn_ratio = np.round(len(mci_data)/len(cn_data))

for i in range(len(cn_data)):
    for j in range(int(cn_ratio)):
        if j == 0:
            vol = Image.open(cn_data[i])
            #vol = CenterCrop.preprocess(vol, 224) # 255.0 * (vol - vol.min()) / (vol.max() - vol.min())
            vol.save(
                CNRes + "\\" + cn_filepaths[i] + "sample" + str(j) + '.png')
        else:
            vol = Image.open(cn_data[i])
            #vol = CenterCrop.preprocess(vol, 224)
            angle = 90
            vol = vol.rotate(angle, expand=True)
            vol.save(
                CNRes + "\\" + cn_filepaths[i] + "sample" + str(j) + '.png')

ad_ratio = np.round(len(mci_data)/len(ad_data))
for i in range(len(ad_data)):
    for j in range(int(2)):
        if j == 0:
            vol = Image.open(ad_data[i])
            #vol = CenterCrop.preprocess(vol, 224)# 255.0 * (vol - vol.min()) / (vol.max() - vol.min())
            vol.save(
                ADRes + "\\" + ad_filepaths[i] + "sample" + str(j) + '.png')
        elif j == 1:
            vol = Image.open(ad_data[i])
            #vol = CenterCrop.preprocess(vol, 224)
            angle = 90
            vol = vol.rotate(angle, expand=True)
            vol.save(
                ADRes + "\\" + ad_filepaths[i] + "sample" + str(j) + '.png')
        elif j == 2:
            vol = Image.open(ad_data[i])
            #vol = CenterCrop.preprocess(vol, 224)
            angle = 180
            vol = vol.rotate(angle, expand=True)
            vol.save(
                ADRes + "\\" + ad_filepaths[i] + "sample" + str(j) + '.png')
        elif j == 3:
            vol = Image.open(ad_data[i])
            #vol = CenterCrop.preprocess(vol, 224)
            angle = 270
            vol = vol.rotate(angle, expand=True)
            vol.save(
                ADRes + "\\" + ad_filepaths[i] + "sample" + str(j) + '.png')
