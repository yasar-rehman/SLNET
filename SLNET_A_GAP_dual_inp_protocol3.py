from __future__ import print_function
# from __future__ import absolute_import
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.models import Sequential, Model, model_from_yaml
from keras.utils import plot_model
from keras.layers import merge, Dense, Dropout, Flatten, concatenate, add, Concatenate, subtract, average, dot
import numpy as np
import scipy
import sys
import os
import argparse
from random import randint, uniform
import time
import matplotlib.pyplot as plt
from keras.losses import sparse_categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as t
import random
import cv2
import keras.backend as K
import tensorflow as tf
import pandas as pd
from keras_preprocessing import image
import random as rn
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)


# -----------------------------------------------------------------------------------------------
# import the essential functions required for computation
# sys.path.insert(0, os.path.expanduser('~//CNN_networks'))
# sys.export PYTHONPATH=/home/yaurehman2/PycharmProjects/face_anti_sp_newidea

print(sys.path)
from  cnn_networks.VGG16_A_GAP_dual_inp import  cnn_hybrid_color_single
from ess_func import read_pairs, sample_people, prewhiten, store_loss, hog_to_tensor, custom_loss


# -----------------------------------------------------------------------------------------------

def main(args):
    # set the image parameters
    img_rows = args.img_rows
    img_cols = args.img_cols
    img_dim_color = args.img_channels
    # mix_prop = 1.0                                                    # set the value of the mixing proportion

    #############################################################################################################
    ##################################  DEFINING MODEL  ##########################################################
    ##############################################################################################################
    model_alex = cnn_hybrid_color_single(img_rows, img_cols, img_dim_color)  # load the model

    # model_final = Model(model_alex.input, model_alex.output)  # specify the input and output of the model
    model_final = model_alex
    print(model_final.summary())  # print the model summary

    plot_model(model_final, to_file='./NIN_hybrid_bin_resnet_1x1-class', show_shapes=True)  # save the model summary as a png file


    lr = args.learning_rate  # set the learning rate

    # set the optimizer
    optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9)

    # model compilation
    model_final.compile(optimizer=optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

    # print the value of the learning rate
    print(K.get_value(optimizer.lr))

    # --------------------------------------------------
    #############################################################################################################
    ########################## GETTING TRAINING DATA AND TESTING DATA  ##########################################
    ##############################################################################################################

    # get the training data by calling the pairs function
    # read the training data

    train_pairs_r, training_data_r, training_label_r = read_pairs(args.tr_img_lab_r)
    train_pairs_l, training_data_l, training_label_l = read_pairs(args.tr_img_lab_l)

    assert len(training_data_r) == len(training_data_l)

    # combine the left and right image in the training data to make a X x Y x 6 tensor
    training_data = []
    for i in range(len(training_data_r)):
        # define the stereo pair
        stereo_pair = [training_data_r[i], training_data_l[i]]
        training_data.append(stereo_pair)

    batch_num = 0

    # initialize the live samples and fake samples
    live_samples_ub = 0
    attack_samples_ub = 0

    live_samples = []
    live_labels = []
    attack_samples = []
    attack_labels = []

    # separate the live samples and fake samples to balance the both classes, i.e. live class and fake class
    assert len(training_label_r) == len(training_label_l)

    for i in range(len(training_data)):
        if training_label_r[i] == 0:
            live_samples.append(training_data[i])
            live_labels.append(training_label_r[i])

            live_samples_ub += 1
        elif (training_label_r[i] == 1) | (training_label_r[i] == 3) | (training_label_r[i] == 4): # cut photo attack removed
            attack_samples.append(training_data[i])
            attack_labels.append(training_label_r[i])

            attack_samples_ub += 1


    print("Live samples are %g ,\t attack samples are %g" % (live_samples_ub, attack_samples_ub))

    # compute the difference; the live samples are always less than the fake samples in our case
    diff = 0
    if live_samples_ub < attack_samples_ub:
        # compute the ratio
        diff = np.int(attack_samples_ub / live_samples_ub)
        print("The difference is :%g " % (diff))
    else:
        ValueError("The fake samples are less than then live samples")

    # number of times the dataset has to be copied:
    live_samples_b = live_samples
    live_labels_b = live_labels
    for i in range(diff - 1):
        # print("length before balancing: %g" %len(live_samples_b))
        sl_copy = live_samples.copy()
        ll_copy = live_labels.copy()

        live_samples_b = live_samples_b + sl_copy
        live_labels_b = live_labels_b + ll_copy
        # print("length after balancing: %g" % len(live_samples_b))

    # balanced data
    training_data_balanced = live_samples_b + attack_samples
    training_label_balanced = live_labels_b + attack_labels

    print("Balanced data samples: %g" % len(training_data_balanced))

    # get the length of the training data
    len_tr = len(training_data_balanced)

    # get the number equal to the length of the training data
    indices_tr = np.arange(len_tr)
    np.random.shuffle(indices_tr)


    # initialize the image counter
    images_read = 0
    train_img_data_r = []
    train_img_data_l = []

    for i in indices_tr:
        if training_label_balanced[i] > 0:
            training_label_balanced[i] = 1

        train_img_data_r.append([training_data_balanced[i][0], training_label_balanced[i]]) # read the right image
        train_img_data_l.append([training_data_balanced[i][1], training_label_balanced[i]]) # read the left image

        # print(training_data_balanced[i][1])
        # cv2.imshow('img1', cv2.imread(training_data_balanced[i][0]))
        # cv2.waitKey()
        # cv2.imshow('img2', cv2.imread(training_data_balanced[i][1]))
        # cv2.waitKey()

        images_read += 1
        sys.stdout.write('train images read = {0}\r'.format(images_read))
        sys.stdout.flush()

    ############################################################################################################

    # read the test data
    test_pairs, test_data_r, test_labels_r = read_pairs(args.tst_img_lab_r)
    test_pairs, test_data_l, test_labels_l = read_pairs(args.tst_img_lab_l)

    assert len(test_data_r) == len(test_data_l)

    # combine the left and right image in the training data to make a X x Y x 6 tensor
    test_data = []
    for i in range(len(test_data_r)):
        # define the stereo pair
        stereo_pair_t = [test_data_r[i], test_data_l[i]]
        test_data.append(stereo_pair_t)

    test_labels = test_labels_r

    images_read = 0

    # get the length of the training data
    len_test = len(test_data)

    # get the number equal to the length of the training data
    indices_test = np.arange(len_test)

    test_img_data_r = []
    test_img_data_l = []


    for i in indices_test:

        if test_labels[i] > 0:
            test_labels[i] = 1

        test_img_data_r.append([test_data[i][0], test_labels[i]]) # read the right test image
        test_img_data_l.append([test_data[i][1], test_labels[i]]) # red the left test image
        images_read += 1
        sys.stdout.write('test images read = {0}\r'.format(images_read))
        sys.stdout.flush()

    #####################################################################################################
    # make all the data in panda data frame format
    train_df_r = pd.DataFrame(train_img_data_r)
    train_df_r.columns = ['id_r', 'label']

    train_df_l = pd.DataFrame(train_img_data_l)
    train_df_l.columns = ['id_l', 'label']

    test_df_r = pd.DataFrame(test_img_data_r)
    test_df_r.columns = ['id_r', 'label']

    test_df_l = pd.DataFrame(test_img_data_l)
    test_df_l.columns = ['id_l', 'label']

    ########################################################################################################333

    datagen = image.ImageDataGenerator()

    train_generator_r = datagen.flow_from_dataframe(
        dataframe=train_df_r,
        directory=None,
        x_col='id_r',
        y_col='label',
        has_ext=True,
        batch_size=args.batch_size,
        seed=42,
        shuffle=True,
        class_mode="sparse",
        target_size=(args.img_rows, args.img_cols),
        color_mode='grayscale',
        interpolation='nearest',
        drop_duplicates=False
    )

    train_generator_l = datagen.flow_from_dataframe(
        dataframe=train_df_l,
        directory=None,
        x_col='id_l',
        y_col='label',
        has_ext=True,
        batch_size=args.batch_size,
        seed=42,
        shuffle=True,
        class_mode="sparse",
        target_size=(args.img_rows, args.img_cols),
        color_mode='grayscale',
        interpolation='nearest',
        drop_duplicates=False
    )


    test_datagen = image.ImageDataGenerator()

    test_generator_r = test_datagen.flow_from_dataframe(
        dataframe=test_df_r,
        directory=None,
        x_col='id_r',
        y_col='label',
        has_ext=True,
        batch_size=args.batch_size,
        seed=42,
        shuffle=False,
        class_mode="sparse",
        target_size=(args.img_rows, args.img_cols),
        color_mode='grayscale',
        interpolation='nearest'
    )

    test_generator_l= test_datagen.flow_from_dataframe(
        dataframe=test_df_l,
        directory=None,
        x_col='id_l',
        y_col='label',
        has_ext=True,
        batch_size=args.batch_size,
        seed=42,
        shuffle=False,
        class_mode="sparse",
        target_size=(args.img_rows, args.img_cols),
        color_mode='grayscale',
        interpolation='nearest'
    )
    #############################################################################################################
    batch_num = 0
    while batch_num < args.max_epochs:

        start_time = time.time()  # initialize the clock
        acc = []
        loss = []
        sub_count = 0

        total_batch = train_generator_r.n // train_generator_r.batch_size

        for i in range(train_generator_r.n // train_generator_r.batch_size):
            x1, y1 = next(train_generator_r)
            x2, y2 = next(train_generator_l)

            # only for DP-3D for comparison
            # disparity_final = []
            #
            # for j in range(x1.shape[0]):
            #     img1 = np.asarray(x1[j])
            #     # img1 = cv2.resize(img1, (img_rows, img_cols),
            #     #                                 interpolation=cv2.INTER_AREA)
            #
            #     img2 = np.asarray(x2[j])
            #     # img2 = cv2.resize(img2, (img_rows, img_cols),
            #     #                                 interpolation=cv2.INTER_AREA)
            #     #
            #     disparity = cv2.subtract(img1, img2)
            #
            #     der_k = np.asarray([[1.0, 2.0, 1.0],
            #                         [0.0, 0.0, 0.0],
            #                         [-1.0, -2.0, -1.0]])
            #
            #     der = cv2.filter2D(img1, -1, kernel=der_k)
            #
            #     disparity_f = disparity / (der + 0.005)
            #
            #     disparity_final.append(disparity_f)
            #
            # # disparity_final = np.asarray(disparity_final).astype('float32')
            # disparity_final = np.expand_dims(np.asarray(disparity_final).astype('float32'), axis=-1)


            x1 = x1.astype('float32') / 255
            x2 = x2.astype('float32') / 255

            y = y1

            tr_acc1 = model_final.fit([x1,x2],
                                       y,
                                      epochs=1,
                                      verbose=0)
            acc.append(tr_acc1.history['acc'][0])
            loss.append(tr_acc1.history['loss'][0])

            sub_count += 1
            sys.stdout.write('batch_count = {0} of {1} \r'.format(sub_count, total_batch))
            sys.stdout.flush()

        train_acc = np.sum(np.asarray(acc)) * 100 / (train_generator_r.n // train_generator_r.batch_size)
        train_loss = np.sum(np.asarray(loss)) * 100 / (train_generator_r.n // train_generator_r.batch_size)

        print('training_acc: {0} \t training_loss: {1}'.format(train_acc, train_loss))

        print('______________________________________________________________________')
        print('Running the evaluations')

        test_acc = []
        test_loss = []
        sub_count = 0

        for i in range(test_generator_r.n // test_generator_r.batch_size):
            x1, y1 = next(test_generator_r)
            x2, y2 = next(test_generator_l)

            # only for DP-3D for comparison
            # disparity_final = []
            #
            # for j in range(x1.shape[0]):
            #     img1 = np.asarray(x1[j])
            #     # img1 = cv2.resize(img1, (img_rows, img_cols),
            #     #                                 interpolation=cv2.INTER_AREA)
            #
            #     img2 = np.asarray(x2[j])
            #     # img2 = cv2.resize(img2, (img_rows, img_cols),
            #     #                                 interpolation=cv2.INTER_AREA)
            #     #
            #     disparity = cv2.subtract(img1, img2)
            #
            #     der_k = np.asarray([[1.0, 2.0, 1.0],
            #                         [0.0, 0.0, 0.0],
            #                         [-1.0, -2.0, -1.0]])
            #
            #     der = cv2.filter2D(img1, -1, kernel=der_k)
            #
            #     disparity_f = disparity / (der + 0.005)
            #
            #     disparity_final.append(disparity_f)
            #
            # # disparity_final = np.asarray(disparity_final).astype('float32')
            # disparity_final = np.expand_dims(np.asarray(disparity_final).astype('float32'), axis=-1)


            x1 = x1.astype('float32') / 255
            x2 = x2.astype('float32') /255

            y = y1

            tst_loss, tst_acc1 = model_final.evaluate([x1,x2],
                                                      y,
                                                      verbose=0)
            test_acc.append(tst_acc1)
            test_loss.append(tst_loss)
            sub_count += 1
            sys.stdout.write('epoch_count = {0}\r'.format(sub_count))
            sys.stdout.flush()

        test_acc = np.sum(np.asarray(test_acc)) * 100 / (test_generator_r.n // test_generator_r.batch_size)
        test_loss = np.sum(np.asarray(test_loss)) * 100 / (test_generator_r.n // test_generator_r.batch_size)

        print('test_acc: {0} \t test_loss: {1}'.format(test_acc, test_loss))

        batch_num += 1

        # **********************************************************************************************
        # learning rate schedule update: if learning is done using a single learning give the batch_num below a
        # high value
        if (batch_num == 3) | (batch_num == 5) | (batch_num == 7):
            lr = 0.1 * lr
            K.set_value(optimizer.lr, lr)
            print(K.get_value(optimizer.lr))

        # ************************************************************************************************
        # -----------------------------------------------------------------------------------------------

        end_time = time.time() - start_time

        print("Total time taken %f :" % end_time)

        model_final.save_weights(
            '/Documents/stereo_face_liveness/stereo_ckpt/Conventional/' + 'dual_grayscale_input_revised_protocol_3_'+ str(args.max_epochs) + '.h5')



def parser_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--tr_img_lab_r', type=str,
                        help='directory from where to get the training paths and ground truth',
                        default='/Documents/Newwork/stereo_face_new_multi-class/train_right_mt_context.txt')

    parser.add_argument('--tr_img_lab_l', type=str,
                        help='directory from where to get the training paths and ground truth',
                        default='/Documents/Newwork/stereo_face_new_multi-class/train_left_mt_context.txt')


    parser.add_argument('--tst_img_lab_r', type=str,
                        help='direcotry where test iamges are stored ',
                        default='/Documents/Newwork/stereo_face_new_multi-class/test_right_mt_context.txt')

    parser.add_argument('--tst_img_lab_l', type=str,
                        help='direcotry where test iamges are stored ',
                        default='/Documents/Newwork/stereo_face_new_multi-class/test_left_mt_context.txt')



    # """**************************************************************************************************************"""

    """Specify the parameters for the CNN Net"""

    parser.add_argument('--batch_size', type=int,
                        help='input batch size to the network', default=32)

    parser.add_argument('--test_batch_size', type=int,
                        help='input test batch size to the network', default=12000)

    parser.add_argument('--max_epochs', type=int,
                        help='maximum number of epochs for training', default=10)

    parser.add_argument('--epoch_batch', type=int,
                        help='Maximum epoch per batch per iteration', default=12000)

    # """**************************************************************************************************************"""

    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate', default=0.01)

    parser.add_argument('--data_augmentation', type=str,
                        help='wheather to include data augmentation or not', default=False)

    parser.add_argument('--img_rows', type=int,
                        help='image height', default=120)

    parser.add_argument('--img_cols', type=int,
                        help='image width', default=120)

    parser.add_argument('--img_channels', type=int,
                        help='number of input channels in an image', default=1)

    parser.add_argument('--epoch_flag', type=int,
                        help='determine when to change the learning rate', default=1)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parser_arguments(sys.argv[1:]))



