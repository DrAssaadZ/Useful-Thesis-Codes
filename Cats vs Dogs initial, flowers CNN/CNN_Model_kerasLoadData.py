'''
CNN implementation of the cats-dogs dataset with keras loading data method
the code works perfectly
'''


import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib


training_data_dir = pathlib.Path('../dataset/training_set/')
testing_data_dir = pathlib.Path('../dataset/test_set/')

training_image_count = len(list(training_data_dir.glob('*/*.jpg')))
testing_image_count = len(list(testing_data_dir.glob('*/*.jpg')))

training_CLASS_NAMES = np.array([item.name for item in training_data_dir.glob('*') if item.name != "LICENSE.txt"])
testing_CLASS_NAMES = np.array([item.name for item in testing_data_dir.glob('*') if item.name != "LICENSE.txt"])

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
IMG_HEIGHT = 64
IMG_WIDTH = 64
STEPS_PER_EPOCH = np.ceil(training_image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(training_data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes=list(training_CLASS_NAMES))

test_data_gen = image_generator.flow_from_directory(directory=str(testing_data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes=list(testing_CLASS_NAMES))


training_iters = 10000
learning_rate = 0.001
n_classes = 2

# both placeholders are of type float
x = tf.placeholder("float", [None, 64, 64, 3])
y = tf.placeholder("float", [None, n_classes])


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


weights = {
    'W1': tf.get_variable('W1', shape=(3, 3, 3, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'W2': tf.get_variable('W2', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
    # 'W21': tf.get_variable('W21', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
    # 'W22': tf.get_variable('W22', shape=(3, 3, 32, 16), initializer=tf.contrib.layers.xavier_initializer()),
    'W3': tf.get_variable('W3', shape=(16*16*32, 128), initializer=tf.contrib.layers.xavier_initializer()),
    'W4': tf.get_variable('W4', shape=(128, n_classes), initializer=tf.contrib.layers.xavier_initializer()),
    # 'W5': tf.get_variable('W5', shape=(256, n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'B1': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'B2': tf.get_variable('B2', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    # 'B21': tf.get_variable('B21', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    # 'B22': tf.get_variable('B22', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
    'B3': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'B4': tf.get_variable('B4', shape=(2), initializer=tf.contrib.layers.xavier_initializer()),
    # 'B5': tf.get_variable('B5', shape=(2), initializer=tf.contrib.layers.xavier_initializer()),
}


def conv_net(x, weights, biases):

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['W1'], biases['B1'])
    # conv1 = tf.nn.dropout(conv1, 0.8)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['W2'], biases['B2'])
    # conv2 = tf.nn.dropout(conv2, 0.8)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    # conv3 = conv2d(conv2, weights['W21'], biases['B21'])
    #
    # conv4 = conv2d(conv3, weights['W22'], biases['B22'])

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, 16*16*32])
    fc1 = tf.add(tf.matmul(fc1, weights['W3']), biases['B3'])
    fc1 = tf.nn.sigmoid(fc1)

    # fc2 = tf.add(tf.matmul(fc1, weights['W4']), biases['B4'])
    # fc2 = tf.nn.relu(fc2)

    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['W4']), biases['B4'])

    return out


pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image.
# and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# for visualization in tensorboard
tf.summary.scalar("Training Accuracy", accuracy)
summary_op = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    validation_loss = []
    train_accuracy = []
    test_accuracy = []
    test_acc_list = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        train_image_batch, train_label_batch = next(train_data_gen)

        # Run optimization op (backprop).
        # Calculate batch loss and accuracy
        opt = sess.run(optimizer, feed_dict={x: train_image_batch, y: train_label_batch})
        # added train summary for visualization(tensorboard)
        train_summary, loss, acc = sess.run([summary_op, cost, accuracy], feed_dict={x: train_image_batch, y: train_label_batch})
        summary_writer.add_summary(train_summary, i)
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")
        train_loss.append(loss)
        train_accuracy.append(acc)

        if i in range(0, 8000, 250):
            if i != 0:
                for j in range(2000 // BATCH_SIZE):
                    test_image_batch, test_label_batch = next(test_data_gen)
                    # Calculate accuracy for all 10000 mnist test images
                    valid_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_image_batch, y: test_label_batch})
                    test_acc_list.append(valid_acc)

                    validation_loss.append(valid_loss)

                test_accuracy.append(np.mean(test_acc_list))

        # print('batch accuracy list : ', test_acc_list)
        print("--------------------------")
        print("Testing Accuracy:", test_accuracy)

    summary_writer.close()







