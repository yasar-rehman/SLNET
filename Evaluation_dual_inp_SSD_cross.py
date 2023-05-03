from keras.models import Sequential, Model
from keras.utils import plot_model
import numpy as np
import scipy
import sys
import argparse
from random import randint, uniform
import time
import matplotlib.pyplot as plt
import tensorflow as t
import random
import cv2
import dlib
from imutils.face_utils import rect_to_bb, shape_to_np
from keras_preprocessing.image import  load_img
from PIL import ImageFilter
# -----------------------------------------------------------------------------------------------
# import the essential functions required for computation
from  cnn_networks.SLNET_A_GAP_disp_dual_inp_A  import cnn_hybrid_color_single
from ess_func import read_pairs, sample_people, prewhiten, store_loss, hog_to_tensor, custom_loss
import os

detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier('/opencv/opencv-3.3.0/data/haarcascades_cuda/haarcascade_frontalface_default.xml')

def format(value):
    return "%.3f" % value

def store_predictions(file_path, prediction, label):
    with open(file_path,'a+') as f:
        f.write("%10.4e \t %10.4e \t %10.4e \t %10.4e" %
                (prediction[0][0], prediction[0][1], prediction[0][2], prediction[0][3]))
        f.write("\t")
        f.write(str(label))
        f.write("\n")

def store_predictions_binary(file_path, prediction, label):
    with open(file_path,'a+') as f:
        f.write("%10.4e \t %10.4e" %
                (prediction[0][0], prediction[0][1]))
        f.write("\t")
        f.write(str(label))
        f.write("\n")



accuracy = []

def main(args):
    nrf_fp = 0
    img_rows = args.img_rows
    img_cols = args.img_cols
    img_dim_color = args.img_channels

    model_alex =  cnn_hybrid_color_single(img_rows, img_cols, img_dim_color)
    model_final = model_alex
    # model_final = Model(model_alex.input, model_alex.output)
    model_final.load_weights(args.weights_path)

    test_pairs, test_data_r, test_labels_r = read_pairs(args.tst_img_lab_r)  # read the test data
    test_pairs, test_data_l, test_labels_l = read_pairs(args.tst_img_lab_l)  # read the test data


    assert  len(test_data_r) == len(test_data_l)
    assert  len(test_labels_r) == len(test_labels_l)

    # Generating teh test samples
    images_sampled = []  # images to be sampled from the data-set
    labels_sampled = []  # sampled labels

    len_test = len(test_data_r)  # get the length of the test data
    indices_test = np.arange(len_test)  # get the number equal to the length of the training data
    images_read = 0
    for i in indices_test:
        # read the corresponding image from the data set
        image_data_r = load_img(test_data_r[i],
                              grayscale=False,
                              color_mode='grayscale',
                              target_size=(args.img_rows, img_cols),
                              interpolation='nearest')  # read the corresponding image from the data set

        image_data_l = load_img(test_data_l[i],
                                grayscale=False,
                                color_mode='grayscale',
                                target_size=(args.img_rows, img_cols),
                                interpolation='nearest')  # read the corresponding image from the data set

        # add a guassian blur
        # image_data_r = image_data_r.filter(ImageFilter.GaussianBlur(radius=2.5))
        # image_data_l = image_data_l.filter(ImageFilter.GaussianBlur(radius=2.5))



        ######################################################################################################3
        # only for disparity input
        # img1 = np.asarray(image_data_r).astype(np.float32)
        # # img1 = cv2.resize(img1, (img_rows, img_cols),
        # #                                 interpolation=cv2.INTER_AREA)
        #
        # img2 = np.asarray(image_data_l).astype(np.float32)
        # # img2 = cv2.resize(img2, (img_rows, img_cols),
        # #                                 interpolation=cv2.INTER_AREA)
        # #
        # disparity = cv2.subtract(img1, img2)
        #
        # der_k = np.asarray([[1.0, 2.0, 1.0],
        #                     [0.0, 0.0, 0.0],
        #                     [-1.0, -2.0, -1.0]])
        #
        # der = cv2.filter2D(img1, -1, kernel=der_k)
        #
        # disparity_f = disparity / (der + 0.005)
        #
        # disparity_final = disparity_f
        #
        # disparity_final = np.asarray(disparity_final).astype('float32')
        # test_img = disparity_final
        ####################################################################################

        image_data_r = np.expand_dims(np.expand_dims(image_data_r, axis=-1), axis=0)
        image_data_l = np.expand_dims(np.expand_dims(image_data_l, axis=-1),axis=0)
        # image_data = np.concatenate((image_data_r,image_data_l), axis=-1)

        # image_data = test_img


        sys.stdout.write('train images read = {0}\r'.format(images_read))
        sys.stdout.flush()
        images_read += 1



        test_imgr = np.asarray(image_data_r).astype('float32') / 255
        test_imgl = np.asarray(image_data_l).astype('float32')/255




        # test_img = np.expand_dims(test_img,axis=0)

        # test_img = np.expand_dims(np.expand_dims(test_img, axis=0),axis=-1)

        test_lab = np.asarray(test_labels_r[i]).astype('int64')

        # flatten the training labels
        test_lab = np.reshape(test_lab, (-1,))

        prediction = model_final.predict([test_imgr, test_imgl])
        # prediction = model_final.predict(test_img)

        #################################################################
        # prediction = model_final.predict([test_img, test_img])
        #################################################################

        store_predictions_binary(args.save_pred, prediction, test_lab)

        correct_prediction = np.equal(np.argmax(prediction, 1), test_lab)
        # correct_prediction = np.float(np.sum(correct_prediction)) * 100 / args.test_batch_size
        accuracy.append(correct_prediction)
        if correct_prediction == 0:
            nrf_fp += 1

    print(sum(accuracy)*100/len(accuracy),nrf_fp)



def parser_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--tst_img_lab_r', type=str,
                        help='direcotry where test iamges are stored ',
                        default='/stereo_face_new_multi-class/test_right_mt_context.txt')

    parser.add_argument('--tst_img_lab_l', type=str,
                        help='direcotry where test iamges are stored ',
                        default='/stereo_face_new_multi-class/test_left_mt_context.txt')


    # """**************************************************************************************************************"""

    """Specify the parameters for the CNN Net"""

    # """**************************************************************************************************************"""
    parser.add_argument('--l2_regul', type=float,
                        help='l2 regularizer', default=5e-4)
    parser.add_argument('--img_rows', type=int,
                        help='image height', default=120)

    parser.add_argument('--img_cols', type=int,
                        help='image width', default=120)

    parser.add_argument('--img_channels', type=int,
                        help='number of input channels in an image', default=1)

    parser.add_argument('--weights_path', type=str,
                        default='/home/yaurehman2/Documents/stereo_face_liveness/stereo_ckpt/Conventional/dual_grayscale_input_revised_protocol_3_10.h5')

    parser.add_argument('--save_pred', type=str,
                        help='saving the the predictions',
                        default='/home/yaurehman2/Documents/stereo_face_liveness/stereo_Analysis/sampling/'
                                'dual_input_protocol3_60x60_revised.txt')


    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parser_arguments(sys.argv[1:]))
