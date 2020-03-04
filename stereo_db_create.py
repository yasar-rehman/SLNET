import numpy as np
import h5py
import cv2
import os
from stereo_db_gen import video_data, get_dataset


img_rows = 700
img_cols = 700
img_dim = 6
# idx_subjects = 50                   # id of the subject, need to be changed for each subject
nrof_vid_per_subject = 15

for i in range(0,51):
    idx_subjects = i
#
# # ___________________________________________________________________________________________________________________
# # *******************************************************************************************************************
#
    st_tr_file_name = os.path.expanduser('~/Documents/Newwork/'
                                         'Stereo_Face_database/subject' + str(idx_subjects))

    dataset_dir_path = os.path.expanduser('~/Documents/Newwork/'
                                          'Stereo_Face_database/subject_' + str(idx_subjects))

    get_dataset(dataset_dir_path, st_tr_file_name)
    #
    # # ____________________________________________________________________________________________________________________
    # # ********************************************************************************************************************
    #
    file_path = st_tr_file_name
    hdf5_path = '/home/yaurehman2/Documents/stereo_face_liveness/'
    file_read = open(file_path, "r")
    left_cam = 'leftcam'

    subjects_left = []
    subjects_right = []
    tr_labels = []
    tr_database = []

    for f in file_read:
            v = f.strip().split("/")
            vid_name = v[7].split(' ', 1)
            vid_descp = vid_name[0].split("_")
            if left_cam in vid_descp:
                subjects_left.append([s for s in f.strip().split(" ", 1)])
            else:
                subjects_right.append([s for s in f.strip().split(" ", 1)])

    subjects_right.sort()
    subjects_left.sort()
    #
    # # ___________________________________________________________________________________________________________________
    # # *******************************************************************************************************************
    #
    tr_dataset_shape = (int(nrof_vid_per_subject)*100, img_rows, img_cols, img_dim)
    tr_labels_shape = (int(nrof_vid_per_subject)*100,)

    File1 = h5py.File(hdf5_path + 'subject_' + str(idx_subjects), mode='w')
    File1.create_dataset("data", tr_dataset_shape, np.uint8)
    File1.create_dataset("labels", tr_labels_shape, np.int8)

    for i in range(len(subjects_left)):
        try:
            # print subjects_left[i][1], subjects_right[i][1]
            if subjects_left[i][1] == subjects_right[i][1]:
                # tr_labels.append(subjects_left[i][1])
                file_path = [subjects_right[i][0], subjects_left[i][0]]
                database_matrix = video_data(file_path, 100, 15, img_rows, img_cols)
                len_db = database_matrix.shape[0]
                File1["data"][i*len_db:(i+1)*len_db, ...] = database_matrix[None]
                lab = [int(s) for s in (len_db*[subjects_left[i][1]])]
                File1["labels"][i*len_db:(i+1)*len_db, ...] = lab
        except ValueError:
            print("The right_cam and left_cam labels must be same")

    File1.close()
# # ___________________________________________________________________________________________________________________
# # *******************************************************************************************************************
# #
# hdf5_file1 = h5py.File(hdf5_path + 'subject_'+str(idx_subjects), "r")
# #
# frames_train = hdf5_file1["train_labels"]
# #
# print [v for v in frames_train]

