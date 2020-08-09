'''
preprocessing file,
loading the data using np
getting interest points
and saving the result as a numpy file
'''

import cv2
import os
import numpy as np
import random
from tqdm import tqdm
import tensorflow as tf
import numpy as np


# dataset path
TRAIN_DIR = 'C:/Users/Ouss/Desktop/corner detectors/catsdogsTrain'
TEST_DIR = 'C:/Users/Ouss/Desktop/corner detectors/catsdogsTest'


'''Labelling the dataset'''


# function that labels the data
def label_img(img):
    word_label = img.split('.')[-3]
    # DIY One hot encoder
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]


# getting the interest points of an image
def get_interest_point(img):
    # getting the image shape
    w, h = img.shape[1], img.shape[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray, 16, 0.01, 48)

    imageIndex = []
    for corner in corners:
        x, y = corner.ravel()
        x = x - 16
        y = y - 16

        # dealing with interest point at the corner of the image
        if x < 0:
            x = 0
        if x + 32 > w:
            x = w - 32

        if y < 0:
            y = 0
        if y + 32 > h:
            y = h - 32

        # cropping image blocks 32x32
        crop_img = img[int(y):int(y + 32), int(x):int(x + 32)]

        imageIndex.append(crop_img)

    # returning the stacked images
    return np.dstack(imageIndex)


# creating train data with interest points
def create_train_data():
    # Creating an empty list where we should store the training data
    # after a little preprocessing of the data
    training_data = []

    # tqdm is only used for interactive loading
    # loading the training data
    for img in tqdm(os.listdir(TRAIN_DIR)):
        # labeling the images
        label = label_img(img)

        path = os.path.join(TRAIN_DIR, img)

        # loading the image from the path and then converting them into
        # greyscale for easier covnet prob
        img = cv2.imread(path)
        # resizing the small images
        if img.shape[0] < 200 or img.shape[1] < 200:
            img = cv2.resize(img, (256, 256))

        training_data.append([get_interest_point(img), np.array(label)])

        # # final step-forming the training data list with numpy array of the images

        # shuffling of the training data to preserve the random state of our data
    random.seed(101)
    random.shuffle(training_data)

    return training_data


# creating test data with interest points
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
        path = os.path.join(TEST_DIR, img)
        img = cv2.imread(path)
        if img.shape[0] < 200 or img.shape[1] < 200:
            img = cv2.resize(img, (256, 256))
        testing_data.append([get_interest_point(img), np.array(label)])

    random.seed(101)
    random.shuffle(testing_data)

    return testing_data


# creating train data without interest points
def create_train_data_without_ip():
    training_data = []

    # tqdm is only used for interactive loading
    # loading the training data
    for img in tqdm(os.listdir(TRAIN_DIR)):
        # labeling the images
        label = label_img(img)

        path = os.path.join(TRAIN_DIR, img)

        # loading the image from the path and then converting them into
        # greyscale for easier covnet prob
        img = cv2.imread(path)
        # resizing the small images

        img = cv2.resize(img, (64, 64))

        training_data.append([np.array(img), np.array(label)])

        # # final step-forming the training data list with numpy array of the images

        # shuffling of the training data to preserve the random state of our data
    random.seed(101)
    random.shuffle(training_data)

    return training_data


# creating train data without interest points
def process_test_data_without_ip():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
        path = os.path.join(TEST_DIR, img)
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        testing_data.append([np.array(img), np.array(label)])

    random.seed(101)
    random.shuffle(testing_data)

    return testing_data


# train data
# train_data = create_train_data()
# cleaned_train_data = [item for item in train_data if item[0].shape[2] == 48]
# np.save('train_data.npy', cleaned_train_data)
# # # test data
# test_data = process_test_data()
# cleaned_test_data = [item for item in test_data if item[0].shape[2] == 48]
# np.save('test_data.npy', cleaned_test_data)

train_data = create_train_data()
# np.save('train_original_data.npy', train_data)
test_data = process_test_data()
# np.save('test_original_data.npy', test_data)