from keras.models import Model
from keras.layers.core import Input, Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer
from keras.utils.visualize_util import plot

from .keras_IndicesPooling import UnPooling2DIndices, MaxPoolingMask2D

import os
import numpy as np
import argparse
import hdf5
from PIL import Image

def CreateSegNet(input_shape, n_labels, kernel=3, pool_size=(2, 2)):

    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_2)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv1)
    conv_2 = BatchNormalization()(conv_1)
    conv_2 = Activation("relu")(conv_2)

    pool_1 = MaxPooling2D(pool_size)(conv_2)
    mask_1 = MaxPoolingMask2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2 = MaxPooling2D(pool_size)(conv_4)
    mask_2 = MaxPoolingMask2D(pool_size)(conv_4)

    conv_5 = Convolution2d(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = Batchnormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2d(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = Batchnormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2d(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = Batchnormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3 = MaxPooling2D(pool_size)(conv_7)
    mask_3 = MaxPoolingMask2D(pool_size)(conv_7)

    conv_8 = Convolution2d(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = Batchnormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2d(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = Batchnormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2d(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = Batchnormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4 = MaxPooling2D(pool_size)(conv_10)
    mask_4 = MaxPoolingMask2D(pool_size)(conv_10)

    conv_11 = Convolution2d(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = Batchnormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2d(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = Batchnormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2d(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = Batchnormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5 = MaxPooling2D(pool_size)(conv_13)
    mask_5 = MaxPoolingMask2D(pool_size)(conv_13)

    # decoder
    unpool_1 = UnPooling2DIndices()([pool_5, mask_5])

    conv_14 = Convolution2d(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = Batchnormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2d(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = Batchnormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2d(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = Batchnormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = UnPooling2DIndices()([conv_16, mask_4])

    conv_17 = Convolution2d(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = Batchnormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2d(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = Batchnormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2d(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = Batchnormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = UnPooling2DIndices()([conv_19, mask_3])

    conv_20 = Convolution2d(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = Batchnormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2d(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = Batchnormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2d(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = Batchnormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = UnPooling2DIndices()([conv_22, mask_2])

    conv_23 = Convolution2d(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = Batchnormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2d(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = Batchnormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = UnPooling2DIndices()([conv_24, mask_1])

    conv_25 = Convolution2d(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = Batchnormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)

    conv_26 = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(conv_26)
    conv_26 = Permute(2, 1)(outputs)
    predictions = Activation("softmax")(conv_26)

    segnet = Model(inputs=inputs, outputs=predictions, name="SegNet")

    return segnet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("")
    parser.add_argument("",default="",help="")
    parser.add_argument("",default = ,help="Input image size")
    parser.add_argument("",type=int,default=,
                      help="")
    parser.add_argument("", type=int,default=,
                      help="")
    args = parser.parse_args()

