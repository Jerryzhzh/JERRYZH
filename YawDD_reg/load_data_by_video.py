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
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for folder in sex_folders:
        if not folder.endswith('jpg'):
            continue
        indiv_files = os.listdir(data_folder + folder)
        size = len(indiv_files)
        train_size = int(size * (1 - split_fact))
        test_files = indiv_files[train_size:]
        print('[INFO]test videos are following:')
        for test_file in test_files:
            print(test_file)
        cnt = 0
        for files in indiv_files:
            cnt += 1
            jpg_files = os.listdir(data_folder + folder + '/' + files)
            if cnt < train_size:
                for jpg_file in jpg_files:
                    if jpg_file.endswith('_1.jpg'):
                        train_labels += [1]
                    else:
                        train_labels += [0]
                    filepath = data_folder + folder + '/' + files + '/' + jpg_file
                    ima = Image.open(filepath)
                    train_images.append(np.array(ima))
            else:
                for jpg_file in jpg_files:
                    if jpg_file.endswith('_1.jpg'):
                        test_labels += [1]
                    else:
                        test_labels += [0]
                    filepath = data_folder + folder + '/' + files + '/' + jpg_file
                    ima = Image.open(filepath)
                    test_images.append(np.array(ima))

    train_images, train_labels = np.array(train_images), np.array(train_labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)
    train_size = train_labels.shape[0]
    test_size = test_labels.shape[0]
    print('the nums of train_images:' + str(train_size))
    print('the nums of test_images:' + str(test_size))
    train_images, test_images, train_labels, test_labels = train_test_split(np.concatenate((train_images, test_images), axis=0),
                                                                            np.concatenate((train_labels, test_labels), axis=0),
                                                                            test_size=split_fact,
                                                                            random_state=5)
    train_set, test_set = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)), tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    print('--------------finish to load dataset--------------------')
    return train_set.shuffle(train_size).batch(batch_size), test_set.batch(batch_size)

if __name__ == '__main__':
    trainset, testset = DataProcess()