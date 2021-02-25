import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from AD_classification.LR import lr_config
from AD_classification.LR.learningratefinder import LearningRateFinder
from tensorflow.keras.callbacks import ModelCheckpoint
from AD_classification.DataLoader.NormalImageWriter import NormalImageWriter
from AD_classification.LR.clr_callback import CyclicLR
import tensorflow as tf
import matplotlib.pyplot as plt
from AD_classification.config.frequentist_config import BS, OUTPUT_PATH, NUM_EPOCHS
from vit_keras import vit
import numpy as np
from AD_classification.config.frequentist_config import AdniDirSkull, DataDir, Normal_PathDir, ImageSize, LabelNum, ImageType, MODEL, slice_num

import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--lr-f", type=int, default=0,
                    help="whether or not to find optimal learning rate")
    ap.add_argument("-s", "--smoothing", type=float, default=0.1,
                    help="amount of label smoothing to be applied")
    args = vars(ap.parse_args())

    # w = NormalImageWriter(AdniDirSkull, DataDir, Normal_PathDir, ImageSize, LabelNum, ImageType)
    # w.write()

    traindatagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, zoom_range=[.8, 1.2], brightness_range=[.8, 1.2],
        featurewise_center=True, featurewise_std_normalization=True)  # 20, .2

    valdatagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, zoom_range=[.8, 1.2], brightness_range=[.8, 1.2],
        featurewise_center=True, featurewise_std_normalization=True)

    train_dir = "C:\\RyeU\\PhD\\Thesis\\Keras\\Dataset_new\\" + str(LabelNum) + "-Class\\" + slice_num +"\\Train"
    valid_dir = "C:\\RyeU\\PhD\\Thesis\\Keras\\Dataset_new\\" + str(LabelNum) + "-Class\\" + slice_num +"\\Validation"

    print("[INFO] Training ...")
    trainGen = traindatagen.flow_from_directory(
        train_dir, shuffle=True, target_size=(ImageSize, ImageSize), batch_size=BS)
    valGen = valdatagen.flow_from_directory(
        valid_dir, shuffle=True, target_size=(ImageSize, ImageSize), batch_size=BS)

    if LabelNum == 2:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    model = vit.vit_b32(
        image_size=ImageSize,
        activation=activation,
        pretrained=True,
        include_top=True,
        pretrained_top=False,
        classes=LabelNum,
        weights="imagenet21k",
        dropout=0.2,
        mlp_dim=3072,  # 3072 hidden layer dimension
        num_heads=12,  # 12 Number of heads in Multi-head Attention layer
        num_layers=12,  # 12 number of transformer
        hidden_size=768  #768 embedding dimension
    )
    filepath = os.path.sep.join([OUTPUT_PATH, MODEL + '-' + str(LabelNum) + "class - " + str(slice_num) + ".hdf5"])
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    stepsize = lr_config.STEP_SIZE * (len(trainGen))
    clr = CyclicLR(mode=lr_config.CLR_METHOD, base_lr=lr_config.MIN_LR, max_lr=lr_config.MAX_LR, step_size=stepsize)

    if LabelNum == 2:
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.15) #label_smoothing=args["smoothing"]
    else:
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15)
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.SGD(lr=lr_config.MIN_LR, momentum=0.9),
        metrics=["accuracy"],
    )

    if args["lr_f"] > 0:
        # initialize the learning rate finder and then train with learning rates ranging from 1e-10 to 1e+1
        print("[INFO] finding learning rate...")
        lrf = LearningRateFinder(model)
        lrf.find(next(iter(trainGen)), 1e-7, 1e+1, epochs=NUM_EPOCHS, stepsPerEpoch=len(trainGen),
                 batchSize=BS)
        lrf.plot_loss()
        plt.savefig(lr_config.LRFIND_PLOT_PATH + "-" + MODEL + "_lr.png")
    else:
        print(len(trainGen))
        print(len(valGen))
        if LabelNum == 2:
            class_weight = {0: 2.36, 1: 1.0}
        elif LabelNum == 3:
            class_weight = {0: 4.2, 1: 1.8, 2: 1.0}
        else:
            class_weight = {0: 3.68, 1: 1.70, 2: 6.09, 3: 1.0, 4: 17.74}
        model.fit(
            trainGen,
            steps_per_epoch=len(trainGen), #np.ceil((len(trainGen) / float(BS)))
            validation_data=valGen,
            validation_steps=len(valGen),
            epochs=NUM_EPOCHS,
            callbacks=[clr, callbacks_list],
            class_weight=class_weight
        )
        # model.save_weights(os.path.join(args.logdir, "vit"))