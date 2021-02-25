import argparse

# Import necessary components to build LeNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow_probability as tfp
from AD_classification.config.frequentist_config import ImageSize, BS, OUTPUT_PATH, k_fold, LabelNum, NUM_EPOCHS


def alexnet_model(img_shape=(ImageSize, ImageSize, 3), n_classes=LabelNum, l2_reg=0., weights=None, kl=None):

    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(tfp.layers.Convolution2DFlipout(
        96, (11, 11), input_shape=img_shape, padding='same',
        kernel_divergence_fn=kl))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(tfp.layers.Convolution2DFlipout
                (256, (5, 5), padding='same', kernel_divergence_fn=kl))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(tfp.layers.Convolution2DFlipout
                (512, (3, 3), padding='same', kernel_divergence_fn=kl))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(tfp.layers.Convolution2DFlipout
                (1024, (3, 3), padding='same', kernel_divergence_fn=kl))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(tfp.layers.Convolution2DFlipout
                (1024, (3, 3), padding='same', kernel_divergence_fn=kl))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(AveragePooling2D())
    # alexnet.add(Flatten())
    # alexnet.add(Dense(4096))
    # alexnet.add(BatchNormalization())
    # alexnet.add(Activation('relu'))
    # alexnet.add(Dropout(0.5))
    #
    # # Layer 7
    # alexnet.add(Dense(4096))
    # alexnet.add(BatchNormalization())
    # alexnet.add(Activation('relu'))
    # alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(tfp.layers.DenseFlipout(n_classes, kernel_divergence_fn=kl))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('sigmoid'))

    if weights is not None:
        alexnet.load_weights(weights)

    return alexnet