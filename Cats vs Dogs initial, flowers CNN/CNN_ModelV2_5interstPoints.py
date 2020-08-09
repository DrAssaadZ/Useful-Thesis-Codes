'''
CNN model with the 5 interest point method
it works
'''

import numpy as np
import tensorflow as tf

train_data = np.load('train_dataV2.npy', allow_pickle=True)
test_data = np.load('test_dataV2.npy', allow_pickle=True)

trainX = np.array([item[0]/255 for item in train_data])
trainY = np.array([item[1] for item in train_data])


testX = np.array([item[0]/255 for item in test_data])
testY = np.array([item[1] for item in test_data])


training_iters = 2000
learning_rate = 0.001
batch_size = 100

n_classes = 2

# both placeholders are of type float
x = tf.placeholder("float", [None, 32, 32, 3])
y = tf.placeholder("float", [None, n_classes])


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


weights = {
    'W1': tf.get_variable('W1', shape=(3, 3, 3, 8), initializer=tf.contrib.layers.xavier_initializer()),
    'W2': tf.get_variable('W2', shape=(3, 3, 8, 8), initializer=tf.contrib.layers.xavier_initializer()),
    'W21': tf.get_variable('W21', shape=(3, 3, 8, 8), initializer=tf.contrib.layers.xavier_initializer()),
    'W22': tf.get_variable('W22', shape=(3, 3, 8, 8), initializer=tf.contrib.layers.xavier_initializer()),
    'W3': tf.get_variable('W3', shape=(32*32*8, 512), initializer=tf.contrib.layers.xavier_initializer()),
    'W4': tf.get_variable('W4', shape=(512, 256), initializer=tf.contrib.layers.xavier_initializer()),
    'W5': tf.get_variable('W5', shape=(256, n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'B1': tf.get_variable('B1', shape=(8), initializer=tf.contrib.layers.xavier_initializer()),
    'B2': tf.get_variable('B2', shape=(8), initializer=tf.contrib.layers.xavier_initializer()),
    'B21': tf.get_variable('B21', shape=(8), initializer=tf.contrib.layers.xavier_initializer()),
    'B22': tf.get_variable('B22', shape=(8), initializer=tf.contrib.layers.xavier_initializer()),
    'B3': tf.get_variable('B3', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'B4': tf.get_variable('B4', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'B5': tf.get_variable('B5', shape=(2), initializer=tf.contrib.layers.xavier_initializer()),
}


def conv_net(x, weights, biases):

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['W1'], biases['B1'])
    conv1 = tf.nn.dropout(conv1, 0.5)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    # conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['W2'], biases['B2'])
    conv2 = tf.nn.dropout(conv2, 0.5)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    # conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['W21'], biases['B21'])

    conv4 = conv2d(conv3, weights['W22'], biases['B22'])

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv4, [-1, 32*32*8])
    fc1 = tf.add(tf.matmul(fc1, weights['W3']), biases['B3'])
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.add(tf.matmul(fc1, weights['W4']), biases['B4'])
    fc2 = tf.nn.relu(fc2)

    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc2, weights['W5']), biases['B5'])

    return out


pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image.
# and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(trainX)//batch_size):
            batch_x = trainX[batch*batch_size:min((batch+1)*batch_size, len(trainX))]
            batch_y = trainY[batch*batch_size:min((batch+1)*batch_size, len(trainY))]
            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: testX, y: testY})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:", "{:.5f}".format(test_acc))
    summary_writer.close()
