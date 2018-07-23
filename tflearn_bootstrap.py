import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

"""
    This script is used to make a simple convolutional neural network
    to use the MNIST dataset, it's composed of:

    - One input data of 28x28x1 pixels
    - One conv2d layer of 32 filters with a 2x2 kernel size
    - One max pooling with 2 strides of 2x2 size
    - One conv2d layer of 64 filters with a 2x2 kernel size
    - One max pooling with 2 strides of 2x2 size
    - One fully connected layer of 1024 neurons
    - One fully connected layer of 10 neurons

    The regression used is the adam optimizer with the cross entropy.
"""

training_images, training_label, test_X, test_Y = mnist.load_data(one_hot=True)

training_images = training_images.reshape([-1, 28, 28, 1])
test_X = test_X.reshape([-1, 28, 28, 1])

convnet = input_data(shape=[None, 28, 28, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)
model.fit({'input': training_images}, {'targets': training_label}, n_epoch=10, validation_set=({'input': test_X}, {'targets': test_Y}), 
    snapshot_step=500, show_metric=True, run_id='mnist')
model.save(model_file='test.model')