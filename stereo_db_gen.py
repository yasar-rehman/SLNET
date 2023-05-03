import numpy as np
import cv2
import os
import dlib
from imutils.face_utils import rect_to_bb, shape_to_np, FaceAligner
from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS
import matplotlib.pyplot as plt


path = '/stereo_face_liveness/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)
fa = FaceAligner(predictor, desiredFaceWidth=700)


def get_dataset(paths, filename):        # this function get the dataset from the given path
    # This function generates a txt file that is used by the video generating function. The txt file contains paths
    # to the video data.
    labels = []
    file1 = open(filename, "w+")
    for path in paths.split(':'):
        if path != '':
            path_exp = os.path.expanduser(path)
            h_subjects = os.listdir(path_exp)       # list all the files and directories inside in the path
            h_subjects.sort()                       # sort all the directories
            nr_of_subjects = len(h_subjects)        # find the length of all the directories
            for i in range(nr_of_subjects):         # count upto the number of directories
                h_subject_num = h_subjects[i]       # select the ith directory

                vid_dir = os.path.join(path_exp, h_subject_num)  # getting the full path of the directory

                if os.path.isdir(vid_dir):                       # check whether the given path is a directory or not
                    nr_of_videos = os.listdir(vid_dir)           # list all the files in that directory
                    nr_of_videos.sort()

                    for index in nr_of_videos:
                        sub_index = index.split('_')

                        if sub_index[2] == 'real':      # first class
                            labels.append([os.path.join(vid_dir, index), 0])
                            file1.write('%s  %d \n' %(os.path.join(vid_dir,index),int(0)))

                        elif sub_index[2] == 'print':   # second class
                            labels.append([os.path.join(vid_dir, index), 1])
                            file1.write('%s  %d \n' % (os.path.join(vid_dir, index), int(1)))

                        elif sub_index[2] == 'cut':     # 3rd class
                            labels.append([os.path.join(vid_dir, index), 2])
                            file1.write('%s  %d \n' % (os.path.join(vid_dir, index), int(2)))
                        elif sub_index[2] == 'mobile':  # 4th class
                            labels.append([os.path.join(vid_dir, index), 3])
                            file1.write('%s  %d \n' % (os.path.join(vid_dir, index), int(3)))
                        elif sub_index[2] == 'tablet':  # 5th class
                            labels.append([os.path.join(vid_dir, index), 4])
                            file1.write('%s  %d \n' % (os.path.join(vid_dir, index), int(4)))
                else:
                    vid_descrip = vid_dir.strip().split('/')
                    sub_index = vid_descrip[7].strip().split('_')
                    if sub_index[2] == 'real':  # first class
                        labels.append([vid_dir, 0])
                        file1.write('%s  %d \n' % (vid_dir,  int(0)))

                    elif sub_index[2] == 'print':  # second class
                        labels.append([vid_dir, 1])
                        file1.write('%s  %d \n' % (vid_dir, int(1)))

                    elif sub_index[2] == 'cut':  # 3rd class
                        labels.append([vid_dir, 2])
                        file1.write('%s  %d \n' % (vid_dir, int(2)))
                    elif sub_index[2] == 'mobile':  # 4th class
                        labels.append([vid_dir, 3])
                        file1.write('%s  %d \n' % (vid_dir, int(3)))
                    elif sub_index[2] == 'tablet':  # 5th class
                        labels.append([vid_dir, 4])
                        file1.write('%s  %d \n' % (vid_dir, int(4)))

    file1.close()

# ---------------------------------------------------------------------------------------------------------------------

def alignImages(im1, im2):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

# ---------------------------------------------------------------------------------------------------------------------
# Function to extract video frames and put them in a corresponding separate directory


def video_data(file_path, frame_length, seed_input, img_rows, img_cols):

    video_matrix_ = []                           # define a video matrix to store video data
    # video_matrix_l = []
    database_matrix = []                         # define the database matrix to store videos along with frames

    capr = cv2.VideoCapture(file_path[0])
    capl = cv2.VideoCapture(file_path[1])


    video_r_length = int(capr.get(cv2.CAP_PROP_FRAME_COUNT))
    video_r_width = int(capr.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_r_height = int(capr.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_r_fps = capr.get(cv2.CAP_PROP_FPS)

    video_l_length = int(capl.get(cv2.CAP_PROP_FRAME_COUNT))
    video_l_width = int(capl.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_l_height = int(capl.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_l_fps = capl.get(cv2.CAP_PROP_FPS)


    print('length of the video = %g ---- height x width = %d x %d --- fps =%g' % (
        video_r_length, video_r_height, video_r_width, video_r_fps))
    print('length of the video = %g ---- height x width = %d x %d --- fps =%g' % (
        video_l_length, video_l_height, video_l_width, video_l_fps))

    counter = 0
    starting_point = 0
    ret_false_count = 0
    while (capr.isOpened()) & (capl.isOpened()):                       # Read all frames of the video
        ret_r, frame_r = capr.read()
        ret_l, frame_l = capl.read()

        if (counter != (video_r_length-ret_false_count)) & (counter != (video_l_length - ret_false_count)):
        # if (counter != frame_length):
            if (ret_r is not False) & (ret_l is not False):

                # imReg, h = alignImages(frame_l, frame_r)

                tensor_translated, check_face_id, _ = facial_tensor_landmarks(frame_r, frame_l, img_rows, img_cols)

                print (counter, check_face_id)

                if check_face_id != 0:
                    video_matrix_.append(tensor_translated)
                    # video_matrix_l.append(tensor_translated[:,:,3:6])
                    counter += 1
                else:
                    ret_false_count += 1
            else:
                ret_false_count += 1
                break
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capr.release()
    capl.release()

    np.random.seed(seed_input)
    if len(video_matrix_)>10:
        dummy = np.random.randint(0, len(video_matrix_), len(video_matrix_))
        print (dummy)
        counter_b = 0
        for i in dummy:
            selected_frames = video_matrix_[i]
            # selected_l_frames = video_matrix_l[i]
            # ----------------------------------------------------------------------------------------------------
            # print selected_l_frames.dtype
            # # for checking the output detected face. It can be commented
            # comb_frames = np.concatenate((selected_r_frames,selected_l_frames), axis=1)
            # cv2.imshow('min_frame', comb_frames)
            # cv2.waitKey()
            # -----------------------------------------------------------------------------------------------------
            # tensor_translated, flg = facial_tensor_landmarks(selected_r_frames, selected_l_frames, img_rows, img_cols)


            database_matrix.append(selected_frames)
            counter_b += 1
            print (np.asarray(database_matrix).shape)
            if counter_b == frame_length:
                break
            # -----------------------------------------------------------------------------------------------------

            # print tensor_translated.shape
            # comb_frames = np.concatenate((tensor_translated[:, :, 0:3], tensor_translated[:, :, 3:6]), axis=1)
            # cv2.imshow('min_frame', comb_frames)
            # cv2.imshow(tensor_translated[:, :, 0:6:2])
            # cv2.waitKey()
            # ------------------------------------------------------------------------------------------------------
    return np.asarray(database_matrix)


def facial_tensor_landmarks(frame_r, frame_l, img_rows, img_cols):

    img_list = [frame_r, frame_l]
    right_Eye_translate = []
    bounding_boxes = []
    for img in img_list:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img

        rects = detector(img)                        # detect the face region
        # print(rects)
        for rect in rects:
            bounding_boxes.append(rects)
            # (x, y, w, h) = rect_to_bb(rect)

            # face_area = cv2.resize(img[y:y + h - 15, x:x + w - 15], (img_rows, img_cols))
            # cv2.imshow('dummy', face_area)
            # cv2.waitKey()
            land_marks = predictor(img, rect)        # detect the landmarks in the face
            land_marks = shape_to_np(land_marks)            # convert  the landmarks in tuples of x and y

            (rEstart, rEend) = FACIAL_LANDMARKS_IDXS['right_eye']   # get the landmark of right eye
            right_Eye_pts = land_marks[rEstart:rEend]               # get the right eye points and arrange them
            right_Eye_translate.append(right_Eye_pts)               # append the right eye points in the array
    # print (np.asarray(right_Eye_translate).shape)

    if np.asarray(right_Eye_translate).shape[0] == 2:               # condition to set to avoid a single image

        # print right_Eye_translate

        # finding the distance between eyes location and adjusting the translation
        (im1RECX, im1RecY) = (right_Eye_translate[0][0][0], right_Eye_translate[0][0][1])  # select the right most point
        (im2RECX, im2RECY) = (right_Eye_translate[1][0][0], right_Eye_translate[1][0][1])  # select the right most point

        # compute the Euclidean distance
        distx = im1RECX - im2RECX
        disty = im1RecY - im2RECY

        M = np.float32([[1, 0, distx], [0, 1, disty]])  # translation matrix
        # print right_Eye_translate[1][2]
        # pts1 = np.float32([right_Eye_translate[1][0], right_Eye_translate[1][1], right_Eye_translate[1][2]])
        # pts2 = np.float32([right_Eye_translate[0][0], right_Eye_translate[0][1], right_Eye_translate[0][2]])

        # M_1 = cv2.getAffineTransform(pts1, pts2)


        rows, cols,dims = img_list[1].shape
        frame1 = (img_list[1])
        # translate the left image so it the face can overlap on the right image
        dst = cv2.warpAffine(frame1, M, (cols, rows))   # do an affine transform
        # dst = cv2.warpAffine(dst, M_1, (cols, rows))

        # -------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------
        # Just for testing purpose. You can comment this portion if you don't want to do test
        # frame_gray1 = dst
        # rects = detector(frame_gray1)
        #
        # for rect in rects:
        #     land_marks = predictor(frame_gray1, rect)
        #     land_marks = shape_to_np(land_marks)
        # (rEstart, rEend) = FACIAL_LANDMARKS_IDXS['right_eye']
        # right_Eye_pts = land_marks[rEstart:rEend]
        # print right_Eye_pts
        #
        # # updated right eye point after translation of second camera image by dstx and disty
        # (x, y) = right_Eye_pts[0]
        # cv2.circle(frame_r, (x, y), 4, (0, 0, 255), -1)
        #
        # # original right eye point in the first camera image
        # (x1, y1) = right_Eye_translate[0][0]
        # cv2.circle(frame_r, (x1, y1), 4, (0, 255, 0), -1)
        # # original right eye point in the second camera shown in the first camera image
        # (x2, y2) = right_Eye_translate[1][0]
        # cv2.circle(frame_r, (x2, y2), 4, (255, 0, 0), -1)
        #
        # vis = np.concatenate((frame_r, dst, frame_r, frame_l), axis=1)
        # cv2.imshow('output', vis)
        # cv2.waitKey(1)
        # ---------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------
        # # For testing purpose
        # tensor1 = np.zeros((frame_r.shape[0], frame_r.shape[1], 2*frame_r.shape[2]))
        # tensor2 = np.zeros((frame_l.shape[0], frame_l.shape[1], 2*frame_l.shape[2]))
        #
        # tensor1[:, :, 0:3] = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
        # tensor1[:, :, 3:6] = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        #
        # tensor2[:, :, 0:3] = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
        # tensor2[:, :, 3:6] = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
        #
        # plt.figure(1)
        # plt.imshow(tensor1[:, :, 0:6:2])
        # plt.figure(2)
        # plt.imshow(tensor2[:, :, 0:6:2])
        # plt.show()

        # Extract the face area after translation

        translated_images = []
        image_translated = [frame_r, dst]
        # cv2.imshow("window", dst)
        # cv2.waitKey()
        rect = 0
        for i in image_translated:
            # rects = detector(i)
            if len(bounding_boxes) != 2:


                print("The rectangle is empty")
                break
            else:

                for rect in bounding_boxes[0] :

                    (x, y, w, h) = rect_to_bb(rect)

                    if x < 0 or y < 0 or w < 0 or h < 0:
                        break
                    else:
                        face_area = cv2.resize(i[y:y+h, x:x+w], (img_rows, img_cols))
                        translated_images.append(face_area)

        if len(translated_images) == 2:
            tensor_translated = np.uint8(np.zeros((face_area.shape[0], face_area.shape[1], 2*face_area.shape[2])))
            tensor_translated[:, :, 0:3] = translated_images[0]
            tensor_translated[:, :, 3:6] = translated_images[1]

            # testing
            plt.figure(3)

            cv2.imshow('window', cv2.resize(tensor_translated[:, :, 6:0:-2],(255,255)))
            # cv2.waitKey(1)
        # plt.show()
            s = 1
            return tensor_translated, s, bounding_boxes
        else:
            s = 0
            rect = 0
            return [], s, rect
    else:
        s = 0
        rect = 0
        return [], s, rect



def facial_tensor_homography(img_list, img_rows, img_cols):
    tensor1 = []
    rect_st = []
    s = 0
    for i in img_list:
        # detect the face region
        rects = detector(i)

        # only accommodate non-empty bounding boxes
        if rects:
            rect_st.append(rects)

    # since we have two images, we need two bounding boxes, one for each image!
    if len(rect_st) != 2:

        print("The rectangle is empty")

        return tensor1, s
        # break
    else:
        cont = 0
        for i in img_list:
           rect1 = []
           for z in rect_st[cont]:
               rect1.append(z)
               print(z)

           if len(rect1) > 1:
               (x, y, w, h) = rect_to_bb(rect1[0])
           else:
               (x, y, w, h) = rect_to_bb(rect1[0])

           if x < 0 or y < 0 or w < 0 or h < 0:
               print("The rectangle is empty")
               s = 0
               return tensor1, s
           else:
               # if len(rect_st)
               cont += 1
               face_area = cv2.resize(i[y:y + h, x:x + w], (img_rows, img_cols))
               tensor1.append(face_area)

        if len(tensor1) == 2:
            reg_img = np.concatenate((tensor1[0], tensor1[1]), axis=-1)
            cv2.imshow('registered', reg_img[:, :, 6:0:-2])
            # cv2.waitKey(1)
            s = 1
            return reg_img, s
        else:
            s = 0
            return tensor1, s






def check_face(frame_r, frame_l, img_rows, img_cols):
    img_list = [frame_r, frame_l]
    right_Eye_translate = []
    for img in img_list:
        # if len(img.shape) == 3:
        #     frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # else:
        #     frame_gray = img

        rects = detector(img,1)  # detect the face region

        for rect in rects:
            land_marks = predictor(img, rect)  # detect the landmarks in the face
            land_marks = shape_to_np(land_marks)  # convert  the landmarks in tuples of x and y
            (rEstart, rEend) = FACIAL_LANDMARKS_IDXS['right_eye']  # get the landmark of right eye
            right_Eye_pts = land_marks[rEstart:rEend]  # get the right eye points and arrange them
            right_Eye_translate.append(right_Eye_pts)  # append the right eye points in the array
    # print np.asarray(right_Eye_translate).shape
    s = 0
    if np.asarray(right_Eye_translate).shape[0] == 2:  # condition to set if there is a single image

        # print right_Eye_translate

        # finding the distance between eyes location and adjusting the translation
        (im1RECX, im1RecY) = (right_Eye_translate[0][0][0], right_Eye_translate[0][0][1])  # select the right most point
        (im2RECX, im2RECY) = (right_Eye_translate[1][0][0], right_Eye_translate[1][0][1])  # select the right most point

        # compute the Euclidean distance
        distx = im1RECX - im2RECX
        disty = im1RecY - im2RECY

        M = np.float32([[1, 0, distx], [0, 1, disty]])  # translation matrix
        # print right_Eye_translate[1][2]
        # pts1 = np.float32([right_Eye_translate[1][0], right_Eye_translate[1][1], right_Eye_translate[1][2]])
        # pts2 = np.float32([right_Eye_translate[0][0], right_Eye_translate[0][1], right_Eye_translate[0][2]])

        # M_1 = cv2.getAffineTransform(pts1, pts2)


        rows, cols, dims = img_list[1].shape
        frame1 = (img_list[1])
        # translate the left image so it the face can overlap on the right image
        dst = cv2.warpAffine(frame1, M, (cols, rows))  # do an affine transform
        # dst = cv2.warpAffine(dst, M_1, (cols, rows))

        # -------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------
        # Just for testing purpose. You can comment this portion if you don't want to do test
        # frame_gray1 = dst
        # rects = detector(img)
        #
        # for rect in rects:
        #     land_marks = predictor(frame_gray1, rect)
        #     land_marks = shape_to_np(land_marks)
        # (rEstart, rEend) = FACIAL_LANDMARKS_IDXS['right_eye']
        # right_Eye_pts = land_marks[rEstart:rEend]
        # print (right_Eye_pts)
        #
        # # updated right eye point after translation of second camera image by dstx and disty
        # (x, y) = right_Eye_pts[0]
        # cv2.circle(frame_r, (x, y), 4, (0, 0, 255), -1)
        #
        # # original right eye point in the first camera image
        # (x1, y1) = right_Eye_translate[0][0]
        # cv2.circle(frame_r, (x1, y1), 4, (0, 255, 0), -1)
        # # original right eye point in the second camera shown in the first camera image
        # (x2, y2) = right_Eye_translate[1][0]
        # cv2.circle(frame_r, (x2, y2), 4, (255, 0, 0), -1)
        #
        # vis = np.concatenate((frame_r, dst, frame_r, frame_l), axis=1)
        # cv2.imshow('output', vis)
        # cv2.waitKey(1)
        # ---------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------
        # # For testing purpose
        # tensor1 = np.zeros((frame_r.shape[0], frame_r.shape[1], 2*frame_r.shape[2])).astype(np.uint8)
        # tensor2 = np.zeros((frame_l.shape[0], frame_l.shape[1], 2*frame_l.shape[2])).astype(np.uint8)
        #
        # tensor1[:, :, 0:3] = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
        # tensor1[:, :, 3:6] = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        #
        # tensor2[:, :, 0:3] = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
        # tensor2[:, :, 3:6] = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
        #
        # plt.figure(1)
        # plt.imshow(tensor1[:, :, 0:6:2])
        # plt.figure(2)
        # plt.imshow(tensor2[:, :, 0:6:2])
        # plt.show()

        # Extract the face area after translation

        translated_images = []
        image_translated = [frame_r, dst]
        # cv2.imshow("window", dst)
        # cv2.waitKey()
        for i in image_translated:
            # rects = detector(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), 1)
            rects = detector(i,1)
            if not rects:
                print("The rectangle is empty")
                break

            for rect in rects:

                (x, y, w, h) = rect_to_bb(rect)

                face_area = cv2.resize(i[y:y + h, x:x + w], (img_rows, img_cols))
                translated_images.append(face_area)
                # print (np.asarray(translated_images).shape)
        if len(translated_images) == 2:
            tensor_translated = np.uint8(np.zeros((face_area.shape[0], face_area.shape[1], 2 * face_area.shape[2])))
            tensor_translated[:, :, 0:3] = translated_images[0]
            tensor_translated[:, :, 3:6] = translated_images[1]
            s = 1
        else:
            s = 0
            # testing
            # plt.figure(3)
            # cv2.imshow('window', tensor_translated[:, :, 0:6:2])
            # cv2.waitKey(1)
    return s









    # ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# st_tr_file_name = os.path.expanduser('~/Documents/Newwork/st_train_vid_paths_dummy')
# #
# dataset_dir_path = os.path.expanduser('~/Documents/Newwork/Stereo_Face_database')
# #
# get_dataset(dataset_dir_path, st_tr_file_name)
# path = ['/Face_anti_spoofing_cnn_evaluation/StereoFace_database/subject_1/Subject_1_real_rightcam_highdef.flv',
#         '/Face_anti_spoofing_cnn_evaluation/StereoFace_database/subject_1/Subject_1_real_leftcam_highdef.flv']
# database_matrix = video_data(path, 100, 15)
