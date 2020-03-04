from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''Import the necessary libraries for carrying out the necessary computation'''
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D,Flatten, AveragePooling2D, Input, concatenate,add, \
    Lambda, GlobalAveragePooling2D, BatchNormalization, Activation

from keras.regularizers import l2
import tensorflow as tf
from keras import backend as K
from custom_layers import LRN2D
'''Import python and other libraries'''
from custom_layers import disparity

def cnn_hybrid_color_single(img_rows, img_cols, img_dims):
    # img_rows = 120  # compute number of rows in an image
    # img_cols = 120  # compute number of columns in an image
    # img_dims = 3  # compute number of dimensions of an image
    reg_param = 0.0005  # define the regularization parameter



    '''First stack VGG 
    ------------------------'''
    inputs_r = Input(shape=(img_rows, img_cols, img_dims))
    inputs_l = Input(shape=(img_rows, img_cols, img_dims))

    model_conv1 = Convolution2D(8,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                kernel_regularizer=l2(reg_param)
                                )
    feat_r = model_conv1(inputs_r)
    feat_l = model_conv1(inputs_l)

    x1 =  disparity()([feat_r,feat_l])

    x1 = Activation('sigmoid')(x1)



    x= Convolution2D(16,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     kernel_regularizer=l2(reg_param)
                     )(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x1,x])

    # feature mapping layer
    ########################################################################################
    feat1 = Convolution2D(2,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='same',
                          kernel_regularizer=l2(reg_param))(x)
    feat1 = BatchNormalization()(feat1)
    feat1 = Activation('relu')(feat1)
    feat1 = GlobalAveragePooling2D()(feat1)


    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.2)(x)

    '''Second stack VGG
    ------------------------'''
    x = Convolution2D(32,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_regularizer=l2(reg_param)
                      )(x)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)

    x = Convolution2D(32,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_regularizer=l2(reg_param)
                      )(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = concatenate([x1,x])

    # feature mapping layer
    ###############################################################################################
    feat2 = Convolution2D(2,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='same',
                          kernel_regularizer=l2(reg_param))(x)
    feat2 = BatchNormalization()(feat2)
    feat2 = Activation('relu')(feat2)
    feat2 = GlobalAveragePooling2D()(feat2)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.2)(x)

    '''Third stack VGG'''
    '''---------------------------------------------------'''
    x = Convolution2D(64,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_regularizer=l2(reg_param)
                      )(x)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)

    x = Convolution2D(64,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_regularizer=l2(reg_param)
                      )(x1)
    x = BatchNormalization()(x)
    x2 = Activation('relu')(x)

    x = Convolution2D(64,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      kernel_regularizer=l2(reg_param)
                      )(x2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x1,x2,x])

    # feature mapping layer
    ###############################################################################################
    feat3 = Convolution2D(2,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='same',
                          kernel_regularizer=l2(reg_param))(x)
    feat3 = BatchNormalization()(feat3)
    feat3 = Activation('relu')(feat3)
    feat3 = GlobalAveragePooling2D()(feat3)



    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.2)(x)

    '''4th VGG Stack'''
    '''---------------------------------------------------'''
    x = Convolution2D(128,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_regularizer=l2(reg_param)
                      )(x)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)

    x = Convolution2D(128,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_regularizer=l2(reg_param)
                      )(x1)
    x = BatchNormalization()(x)
    x2 = Activation('relu')(x)

    x = Convolution2D(128,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      kernel_regularizer=l2(reg_param)
                      )(x2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x1, x2, x])
    # feature mapping layer
    ###############################################################################################
    feat4 = Convolution2D(2,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='same',
                          kernel_regularizer=l2(reg_param))(x)
    feat4 = BatchNormalization()(feat4)
    feat4 = Activation('relu')(feat4)
    feat4 = GlobalAveragePooling2D()(feat4)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.2)(x)



    '''5th VGG stack'''
    '''---------------------------------------------------'''
    x = Convolution2D(256,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_regularizer=l2(reg_param)
                      )(x)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)

    x = Convolution2D(256,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_regularizer=l2(reg_param)
                      )(x1)
    x = BatchNormalization()(x)
    x2 = Activation('relu')(x)

    x = Convolution2D(256,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding='same',
                      kernel_regularizer=l2(reg_param)
                      )(x2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = concatenate([x1,x2,x])
    featf = Convolution2D(2,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='same',
                          kernel_regularizer=l2(reg_param))(x)
    featf = BatchNormalization()(featf)
    featf = Activation('relu')(featf)
    featf = GlobalAveragePooling2D()(featf)

    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # x = Dropout(0.2)(x)


    '''rescaling net'''
    '''-------------------------------------------------------'''




    c5 = concatenate([feat1, feat2, feat3,feat4, featf], axis=-1)
    #out = GlobalAveragePooling2D()(c5)

    tr_predict = Dense(2)(c5)
    tr_ac = Activation('softmax')(tr_predict)

    model = Model([inputs_r,inputs_l], tr_ac)
    return model


