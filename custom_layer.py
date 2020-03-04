from __future__ import division
from keras.engine import Layer, InputSpec
from keras import initializers
from keras import backend as K
import tensorflow as tf
from keras.layers import Input, Subtract
import numpy as np
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, concatenate, Lambda, GlobalMaxPool2D, Flatten, Embedding, division
import functools

class disparity(Layer):
    def __init__(self, **kwargs):
        super(disparity, self).__init__(**kwargs)

    def build(self, input_shape):
        # input contain two elements (batch x feature vector) and labels

        super(disparity, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # repeat the labels to make it equivalent to the feature vectors
        print(inputs)
        self.feat_vect_r = inputs[0]
        self.feat_vect_l = inputs[1]

        # define a kernel of the shape [filter_height, filter_width, in_channels, channel multiplier_for_depthwise_conv2d]
        der_k = np.asarray([[1.0, 2.0, 1.0],
                            [0.0, 0.0, 0.0],
                            [-1.0, -2.0, -1.0]])
        der_k = np.expand_dims(np.expand_dims(der_k, axis=-1),-1)

        der_kern = np.repeat(der_k, 8, axis=-2)



        kernel = tf.constant(der_kern, dtype=1)   # convert the numpy array into a tensorflow constant


        self.depth_d_3d = self.feat_vect_r -self.feat_vect_l

        der = tf.nn.depthwise_conv2d(self.feat_vect_l, kernel, [1, 1, 1, 1], padding='SAME')

        self.depth_d_3d = self.depth_d_3d / (K.abs(der)+0.5)
