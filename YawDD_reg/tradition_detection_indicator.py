import os
import numpy as np
from PIL import Image
from imutils import face_utils
import imutils
import dlib
import cv2
import tensorflow as tf

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def DataProcess(data_folder='../YawDD dataset/Mirror/'):
    print('--------------start to load dataset--------------------')
    sex_folders = os.listdir(data_folder)
    images = []
    labels = []
    for folder in sex_folders:
        if not folder.endswith('jpg'):
            continue
        indiv_files = os.listdir(data_folder + folder)
        for files in indiv_files:
            jpg_files = os.listdir(data_folder + folder + '/' + files)
            for jpg_file in jpg_files:
                if jpg_file.endswith('_1.jpg'):
                    labels += [1]
                else:
                    labels += [0]
                filepath = data_folder + folder + '/' + files + '/' + jpg_file
                ima = Image.open(filepath)
                images.append(np.array(ima))

    labels = np.array(labels)
    size = labels.shape[0]
    print('the nums of images:' + str(size))
    print('--------------finish to load dataset--------------------')
    return images, labels

def traditioal_model(images):
    MAR_THRESH = 0.6

    (Start, End) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    predict = []
    for image in images:
        frame = imutils.resize(image, width=720)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)

            shape = face_utils.shape_to_np(shape)

            Mouth = shape[Start:End]

            mouth = mouth_aspect_ratio(Mouth)

        if mouth > MAR_THRESH:
            predict.append(1)
        else:
            predict.append(0)
    predict = np.array(predict)
    return predict


images, labels = DataProcess()
predictions = traditioal_model(images)
test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')
test_pre = tf.keras.metrics.Precision(name='test_pre')
test_recall = tf.keras.metrics.Recall(name='test_recall')
test_auc = tf.keras.metrics.AUC(name='test_auc')
test_accuracy(labels, predictions)
test_pre(labels, predictions)
test_recall(labels, predictions)
test_auc(labels, predictions)
template = 'Accuracy: {}, Precision: {}, Recall: {}, Auc: {}'
print(template.format(test_accuracy.result()*100,
                      test_pre.result()*100,
                      test_recall.result()*100,
                      test_auc.result()*100))