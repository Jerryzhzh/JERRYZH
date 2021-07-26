import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

def DataProcess(data_folder='../YawDD dataset/Mirror/',
                split_fact=0.1,
                batch_size=32):
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

    images = np.array(images)
    labels = np.array(labels)
    size = labels.shape[0]
    print('the nums of images:' + str(size))
    print('--------------finish to load dataset--------------------')
    print('--------------start to split dataset--------------------')
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=split_fact, random_state=5)
    train_set, test_set = tf.data.Dataset.from_tensor_slices((train_images, train_labels)), tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    print('--------------finish to split dataset--------------------')
    return train_set.shuffle(size).batch(batch_size), test_set.batch(batch_size)