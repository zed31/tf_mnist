#!/usr/bin/python3

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

"""
    This script is using tflearn instead of Tensorflow.
    thanks to tflearn, you can apply computationnal layers easily than
    by using low-level of tensorflow
"""

class mnist_dataset:

    """
        Class used to store the mnist dataset serialized
    """
    def __init__(self):
        """
            This constructor is used to initialize the training images, training labels as well as
            test images and test labels.
        """
        self.training_images, self.training_labels, self.test_images, self.test_labels = mnist.load_data(one_hot=True)
    
    def reshape(self):
        """
            This function reshape the data coming from the mnist dataset it put -1
            as the first dimentsion as we don't know the number of input we got
            it put then 28 x 28 of the second and third dimension for its pixel number
            and then 1 as the last dimnesion as we have only black and white images
        """
        self.training_images = self.training_images.reshape([-1, 28, 28, 1])
        self.test_images = self.test_images.reshape([-1, 28, 28, 1])
    

class conv_net:
    """
        This class is used to build the convolutionnal neural network
    """
    def __init__(self):
        """
            This constructor initialize the first placeholder for the
            convolutionnal neural network

            this placeholder is set up with a None as first dimension as we don't know how many input
            will goes inside at an Epoch, the second and third dimension is respectively the width and the
            height of the input (28x28) and the last one is the channel colours, which is 1
        """
        self.input_placeholder = input_data(shape=[None, 28, 28, 1], name='Input_placeholder')
    
    def setup_conv_net(self):
        """
            This function setup the convolutionnal network with all the layers
        """
        self.first_conv2d = conv_2d(self.input_placeholder, 32, 2, activation='relu', name='First_conv2D')
        self.first_maxpool = max_pool_2d(self.first_conv2d, 2, name='First_maxpool2D')
        
        self.sec_conv2d = conv_2d(self.first_maxpool, 64, 2, activation='relu', name='Sec_conv2D')
        self.sec_maxpool = max_pool_2d(self.sec_conv2d, 2, name='Sec_maxpool2D')
    
    def setup_fully_connected(self):
        """
            This function set up a fully connected layer of simple neurons

            it also use the dropout to remove some samples during tests and trainings
        """
        self.fully_connected = fully_connected(self.first_maxpool, 1024, activation='relu')
        self.dropout_output = dropout(self.fully_connected, 0.8)
        self.output_layer = fully_connected(self.dropout_output, 10, activation='softmax')
        self.final_output_layer = regression(self.output_layer, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')

dataset = mnist_dataset()
convnet = conv_net()
convnet.setup_conv_net()
convnet.setup_fully_connected()

model = tflearn.DNN(convnet.final_output_layer)
model.fit(
    dataset.training_images, 
    dataset.training_labels, 
    n_epoch=10, 
    validation_set=(dataset.test_images, dataset.test_labels), 
    snapshot_step=500, show_metric=True, run_id='mnist')
#model.save(model_file='mnist_cnn.model')