#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def unpickle_file(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def prepare_to_plot_images(data_batch):
    X = data_batch[b'data']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('uint8')
    return X

def show_images(images, cols = 1, titles = None):
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        print('Treating image:' + str(n) + ' over: ' + str(len(images)))
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def one_hot_encoding(vector, vals=10):
    n = len(vector)
    out = np.zeros((n, vals))
    out[range(n), vector] = 1
    return out


class CifarHelper:
    def __init__(self, batches, test_batches):
        self.i = 0
        self.all_train_batches = batches
        self.test_batch = test_batches

        self.training_images = None
        self.training_labels = None
        
        self.tests_images = None
        self.tests_labels = None

    def set_up_images(self):
        print('Setting up The trainings images and labels')

        self.training_images = np.vstack([d[b'data'] for d in self.all_train_batches])
        training_image_len = len(self.training_images)

        self.training_images = self.training_images.reshape(training_image_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.training_labels = one_hot_encoding(np.hstack([d[b'labels'] for d in self.all_train_batches]), 10)

        print('Setting up The test images and labels')
        self.tests_images = np.vstack([d[b'data'] for d in self.test_batch])
        test_images_length = len(self.tests_images)
        self.tests_images = self.tests_images.reshape(test_images_length, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.tests_labels = one_hot_encoding(np.hstack([d[b'labels'] for d in self.test_batch]), 10)
    
    def next_batch(self, batch_size):
        training_batch = self.training_images[self.i:self.i + batch_size].reshape(100, 32, 32, 3)
        label_batch = self.test_batch[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return training_batch, label_batch

class TFUtils:
    def __init__(self):
        self.set_up_placeholder()

    def set_up_placeholder(self):
        self.placeholder_training_batch = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.placeholder_labels = tf.placeholder(tf.float32, shape=[None, 10])
        self.placeholder_dropout = tf.placeholder(tf.float32)

    def init_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    
    def init_biases(self, shape):
        return tf.Variable(tf.constant(0.1, shape))
    
    def generate_conv2d_layer(self, input_layer, filters_num):
        return tf.nn.conv2d(inputs=input_layer, filters=filters_num, strides=[1, 1, 1, 1], padding='same')
    
    def generate_max_pool_2by2(self, input):
        return tf.nn.max_pool(inputs=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='same')
    
    

CIFAR_DIR = './cifar/'

batch_meta = unpickle_file(file=CIFAR_DIR+'batches.meta')
data_batch1 = unpickle_file(file=CIFAR_DIR+'data_batch_1')
data_batch2 = unpickle_file(file=CIFAR_DIR+'data_batch_2')
data_batch3 = unpickle_file(file=CIFAR_DIR+'data_batch_3')
data_batch4 = unpickle_file(file=CIFAR_DIR+'data_batch_4')
data_batch5 = unpickle_file(file=CIFAR_DIR+'data_batch_5')
test_batch = unpickle_file(file=CIFAR_DIR+'test_batch')

cfHelper = CifarHelper([data_batch1, data_batch2, data_batch3, data_batch4, data_batch5], [test_batch])
cfHelper.set_up_images()


