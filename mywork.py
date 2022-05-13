# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 00:48:22 2022

@author: onyej
"""
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random as r
import tensorflow as tf
from sklearn.utils import shuffle


# read images
def load_images(path_list):
    number_samples = len(path_list)
    Images = []
    for each_path in path_list:
        img = plt.imread(each_path)
        # divided by 255.0
        img = img.reshape(-1, 784) / 255.0
        Images.append(img)
    data = tf.convert_to_tensor(np.array(Images).reshape(number_samples, 784), dtype=tf.float32)
    return data


# load training & testing data
train_data_path = 'train_data'  # Make sure folders and your python script are in the same directory. Otherwise, specify the full path name for each folder.
test_data_path = 'test_data'  # Make sure folders and your python script are in the same directory. Otherwise, specify the full path name for each folder.
train_data_root = pathlib.Path(train_data_path)
test_data_root = pathlib.Path(test_data_path)

# list all training images paths，sort them to make the data and the label aligned
all_training_image_paths = list(train_data_root.glob('*'))
all_training_image_paths = sorted([str(path) for path in all_training_image_paths])

# list all testing images paths，sort them to make the data and the label aligned
all_testing_image_paths = list(test_data_root.glob('*'))
all_testing_image_paths = sorted([str(path) for path in all_testing_image_paths])

# load labels
training_labels = np.loadtxt('labels/train_label.txt',
                             dtype=int)  # Make sure folders and your python script are in the same directory. Otherwise, specify the full path name for each folder.
# convert 1-5 to 0-4 and build one hot vectors
training_labels = tf.reshape(tf.one_hot(training_labels, 10, dtype=tf.float32), (-1, 10))
testing_labels = np.loadtxt('labels/test_label.txt',
                            dtype=int)  # Make sure folders and your python script are in the same directory. Otherwise, specify the full path name for each folder.
testing_labels = tf.reshape(tf.one_hot(testing_labels, 10, dtype=tf.float32), (-1, 10))

# load images
training_set = load_images(all_training_image_paths)
testing_set = load_images(all_testing_image_paths)


i_s = 784
h_1 = 100
h_2 = 100
y_1 = 10
learning_rate = 0.1
batch_size = 50
epoch = 10

    # Weights
W_1a = tf.Variable(tf.random.normal([i_s, h_1], mean=0., stddev=0.01, dtype=tf.float32, seed=1))  # FIRST WEIGHT
W_2a = tf.Variable(tf.random.normal([h_1, h_2], mean=0., stddev=0.01, dtype=tf.float32, seed=2))  # SECOND WEIGHT
W_3a = tf.Variable(tf.random.normal([h_2, y_1], mean=0., stddev=0.01, dtype=tf.float32, seed=3))  # THIRD WEIGHT
    # Weight Biases
W_O1a = tf.Variable(tf.random.normal([h_1], mean=0., stddev=0.01, dtype=tf.float32, seed=1))  # CORRESPONDING WEIGHT BIAS 1
W_O2a = tf.Variable(tf.random.normal([h_2], mean=0., stddev=0.01, dtype=tf.float32, seed=2))  # CORRESPONDING WEIGHT BIAS 2
W_O3a = tf.Variable(tf.random.normal([y_1], mean=0., stddev=0.01, dtype=tf.float32, seed=3))  # CORRESPONDING WEIGHT BIAS 3



def forward_propagation(trainin_set,W_1,W_2,W_3,W_O1,W_O2,W_O3):  # let us call the embedded functions z_1, z_2 and z_3
    z_1 = tf.matmul(trainin_set,W_1) + W_O1
    hidden_1 = tf.nn.relu(z_1)
    z_2 = tf.matmul(hidden_1, W_2) + W_O2
    hidden_2 = tf.nn.relu(z_2)
    z_3 = tf.matmul(hidden_2, W_3) + W_O3
    output = tf.nn.softmax(z_3)
    train_prim = tf.argmax(output, 1)
    return output, z_3, hidden_2, z_2, hidden_1, z_1, train_prim

def loss_function(training_set,training_labels,W_1,W_2,W_3,W_O1,W_O2,W_O3):
    estimate, z_3, hidden_2, z_2, hidden_1, z_1, d = forward_propagation(training_set,W_1,W_2,W_3,W_O1,W_O2,W_O3)
    prod_log = tf.math.multiply(training_labels, tf.math.log(estimate+1e-20))
    var_function = tf.reduce_sum((prod_log), axis=1)
    loss_proper = -tf.reduce_mean(var_function)
    return loss_proper

def Back_propagation(trainin_set,trainin_labels,W_1,W_2,W_3,W_O1,W_O2,W_O3):

    est, z_o3, hid_2, z_o2, hid_1, z_o1, t = forward_propagation(trainin_set,W_1,W_2,W_3,W_O1,W_O2,W_O3)
    output_gradient = -(trainin_labels / est)
    p = tf.eye(y_1, batch_shape=[batch_size])-tf.expand_dims(est, 1)
    q = tf.expand_dims(est * output_gradient, 1)
    c_z3 = tf.squeeze(tf.matmul(q, p))
    w_3_g = tf.matmul(tf.transpose(hid_2), c_z3)
    w_O3_g = tf.reduce_mean(c_z3, axis=0)
    h_2_g = tf.matmul(c_z3, tf.transpose(W_3))
    c_z2 = tf.squeeze(tf.matmul(tf.expand_dims(h_2_g, 1), tf.linalg.diag(tf.cast(z_o2 > 0, tf.float32))))
    w_2_g = tf.reduce_mean(tf.matmul(tf.expand_dims(hid_1, -1), tf.expand_dims(c_z2, 1)), axis=0)
    w_O2_g = tf.reduce_mean(c_z2, axis=0)
    h_1_g = tf.matmul(c_z2, tf.transpose(W_2))
    c_z1 = tf.squeeze(tf.matmul(tf.expand_dims(h_1_g, 1), tf.linalg.diag(tf.cast(z_o1 > 0, tf.float32))))
    w_1_g = tf.reduce_mean(tf.matmul(tf.expand_dims(trainin_set, -1), tf.expand_dims(c_z1, 1)), axis=0)
    w_O1_g = tf.reduce_mean(c_z1, axis=0)
    return w_3_g, w_O3_g, w_2_g, w_O2_g, w_1_g, w_O1_g

def weights_update(trainin_set,trainin_labels,learning_rate,W_1,W_2,W_3,W_O1,W_O2,W_O3):
    q, s, w, t, u, v = Back_propagation(trainin_set,trainin_labels,W_1,W_2,W_3,W_O1,W_O2,W_O3)
    w1_up = (W_1 - learning_rate * u)
    w2_up = (W_2- learning_rate * w)
    w3_up = (W_3- learning_rate * q)
    w10_up = (W_O1-learning_rate * v)
    w20_up = (W_O2- learning_rate * t)
    w30_up = (W_O3- learning_rate * s)
    return w1_up, w2_up, w3_up, w10_up, w20_up, w30_up





def classification_error(y_predicted, y_actual):
    diff_class = np.zeros(10)
    for i in range(len(y_predicted)):
        if y_predicted[i] != y_actual[i]:
            diff_class[y_actual[i]] += 1
    digits2, freq2 = np.unique(y_actual,return_counts=True)
    error = diff_class / freq2
    return error

index_across_y_train = tf.argmax(training_labels, 1)
index_accross_y_test = tf.argmax(testing_labels, 1)


train_error = []
test_error = []
cost_train = []
cost_train_truth = []
for i in range(epoch):
    training_set, training_labels = shuffle(np.array(training_set), np.array(training_labels))
    index_across_y_train = tf.argmax(training_labels, 1)
    output, z_3, hidden_2, z_2, hidden_1, z_1, train_prim = forward_propagation(training_set, W_1a, W_2a, W_3a, W_O1a,
                                                                                W_O2a, W_O3a)
    y_index_grad = np.array(train_prim)
    index_across_y_train = np.array(index_across_y_train)
    initial_error = classification_error(y_index_grad, index_across_y_train)
    #print(initial_error)
    train_error.append(initial_error)
    output2, z_32, hidden_22, z_22, hidden_12, z_12, y_index_final_test = forward_propagation(testing_set, W_1a, W_2a,
                                                                                              W_3a, W_O1a, W_O2a, W_O3a)
    y_index_final_test = np.array(y_index_final_test)
    index_accross_y_test = np.array(index_accross_y_test)
    second_error = classification_error(y_index_final_test, index_accross_y_test)
    #print(second_error)
    test_error.append(second_error)
    loss_proper = loss_function(training_set, training_labels, W_1a, W_2a, W_3a, W_O1a, W_O2a, W_O3a)
    cost_train.append(loss_proper)
    #print("train_loss is:", loss_proper)
    loss_proper_2 = loss_function(testing_set, testing_labels, W_1a, W_2a, W_3a, W_O1a, W_O2a, W_O3a)
    cost_train_truth.append(loss_proper_2)
    #print("test_loss is:", loss_proper_2)
    batch_no = len(training_set) // batch_size
    #print("Epoch is :", i)
    for j in range(batch_no):
        trainset_batch = training_set[j * batch_size: j * batch_size + batch_size]
        trainlabel_batch = training_labels[j * batch_size: j * batch_size + batch_size]
        q, s, w, t, u, v = Back_propagation(trainset_batch , trainlabel_batch, W_1a, W_2a, W_3a, W_O1a, W_O2a, W_O3a)
        W_1a  = (W_1a - learning_rate * u)
        W_2a = (W_2a - learning_rate * w)
        W_3a = (W_3a - learning_rate * q)
        W_O1a = (W_O1a - learning_rate * v)
        W_O2a = (W_O2a - learning_rate * t)
        W_O3a = (W_O3a - learning_rate * s)

mx_weight = [W_1a,W_O1a,W_2a,W_O2a,W_3a,W_O3a]

filehandler = open('nn_parameters.txt', 'wb')
pickle.dump(mx_weight, filehandler, protocol=2)
filehandler.close()


# let us plot the training and testing error
train_error = np.array(train_error)
test_error = np.array(test_error)

for i in range(10):
    plt.plot(train_error[:, i], label="train error")
    plt.plot(test_error[:, i], label="test error")
    plt.legend()
    plt.xlabel('number of Epochs')
    plt.ylabel('Error for digit ' + str(i))
    plt.show()


# Average_error
average_error_train = np.sum(train_error, 1) / 10
average_error_test = np.sum(test_error, 1) / 10
print("")
#print(average_error_train[-1])
#print(average_error_test[-1])

plt.plot(average_error_train, label="Average train error")
plt.plot(average_error_test, label="Average test error")
plt.legend()
plt.xlabel('number of Epochs')
plt.ylabel('average error for digits')
plt.show()

plt.plot(1 - average_error_train, label="Average train accuracy")
plt.plot(1 - average_error_test, label="Average test accuracy")
plt.legend()
plt.xlabel('number of Epochs')
plt.ylabel('average accuracy for digits')
plt.show()

plt.plot(cost_train, label="overall training loss")
plt.plot(cost_train_truth, label="overall testing loss")
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Total loss for digits')
plt.show()













