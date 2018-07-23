#!/usr/bin/python3

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tflearn.datasets.cifar10 as cifar10

"""
    This code is used to show you how to implement a simple cifar10 recognizer using
    the tflearn library.

    Tflearn library is used to create neural network without all the computation needed

    First we load the dataset of the cifar10 by specifying the one hot encoding
"""
(training_images, training_labels), (test_images, test_label) = cifar10.load_data(one_hot=True)

"""
    Next we shuffle the input, to make sure the data will come inside our tensor randomly
"""
(training_images, training_labels) = shuffle(training_images, training_labels)

"""
    In order to use the crossentropy (more precisely the categorical crossentropy) we are
    going to use the categorization function from the tflearn data utils 

    on the one-hot encoded training and test labels
"""
training_labels = to_categorical(training_labels, 10)
test_label = to_categorical(test_label, 10)

"""
    Next things we want to do is adding some image processing under runtime processing
    of our network

    The main idea of applying a zero center data is to put the image at the center
    of the linear plan

    Finally we scale all the samples by applying a standard deviation to it
"""
image_preprocessing = ImagePreprocessing()
image_preprocessing.add_featurewise_zero_center()
image_preprocessing.add_featurewise_stdnorm()

"""
    Once we've done with the pre-processing, we're going to add image augmentation
    this processing applied to the training time only

    We first flipping image randomly to left and right, then we decide to move them from
    90 degrees to make sure the network won't recognize it at the first place
"""
image_augmentation = ImageAugmentation()
image_augmentation.add_random_flip_leftright()
image_augmentation.add_random_rotation(max_angle=25.)

"""
    Then we create our network and our convolutional layer
"""
network_tensor = input_data(shape=[None, 32, 32, 3], 
                            data_preprocessing=image_preprocessing, 
                            data_augmentation=image_augmentation)
network_tensor = conv_2d(incoming=network_tensor, 
                        nb_filter=32, 
                        filter_size=3, 
                        activation='relu')
network_tensor = max_pool_2d(incoming=network_tensor, kernel_size=2)
network_tensor = conv_2d(incoming=network_tensor, 
                        nb_filter=64, 
                        filter_size=3, 
                        activation='relu')
network_tensor = conv_2d(incoming=network_tensor, 
                        nb_filter=64, 
                        filter_size=3, 
                        activation='relu')
network_tensor = max_pool_2d(incoming=network_tensor, kernel_size=2)
network_tensor = fully_connected(incoming=network_tensor, 
                                n_units=512, 
                                activation='relu')
network_tensor = dropout(incoming=network_tensor, 
                        keep_prob=0.5)
network_tensor = fully_connected(incoming=network_tensor, 
                                n_units=10, 
                                activation='softmax')
network_tensor = regression(incoming=network_tensor, 
                            optimizer='adam', 
                            loss='categorical_crossentropy', 
                            learning_rate=0.001)
model = tflearn.DNN(network=network_tensor)
model.fit(training_images, training_labels, 
        n_epoch=50, 
        shuffle=True, 
        validation_set=(test_images, test_label),
        batch_size=96,
        show_metric=True,
        run_id='cifar10_cnn')
