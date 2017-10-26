from keras import backend as K
from keras.layers.convolutional import UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Layer
from keras.layers.merge import Multiply, Concatenate
import numpy as np


class MaxPoolingMask2D(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MaxPoolingMask2D, self).__init__(**kwargs)

    def call(self, x):
        orig = x
        x = MaxPooling2D()(x)
        x = UpSampling2D()(x)
        mask = K.tf.equal(orig, x)
        assert mask.get_shape().as_list() == orig.get_shape().as_list()
        return K.cast(mask, dtype="float32")

    def compute_output_shape(self, input_shape):
        return input_shape

# future implement
"""
class UnPooling2DIndices(Layer):
    def __init__

"""
