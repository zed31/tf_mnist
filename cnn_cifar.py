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
        label_batch = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return training_batch, label_batch

class TFUtils:
    def __init__(self):
        self.set_up_placeholder()

    def set_up_placeholder(self):
        """Setup the placeholder to hold the different shapes of the data
            Here, a shape is the shape of the image (number of pixels x the number of colours)
        """
        self.placeholder_training_batch = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        self.placeholder_labels = tf.placeholder(tf.float32, shape=[None, 10])
        self.placeholder_dropout = tf.placeholder(tf.float32)

    def init_filters(self, shape):
        """Initialize filters with a standard deviation of 0.1
            :param shape: usually the shape of the image
        """
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    
    def init_biases(self, shape):
        """Init the biases according to the defined shape
            :param shape: the shape of the image
        """
        return tf.Variable(tf.constant(0.1, shape=shape))
    
    def generate_conv2d_layer(self, input_layer, filters):
        """Generate a convolutionnal 2D layer according to the defined filters and the input layer
            :param input_layer: The previous layer / placeholder to attach to the new conv2d layer
            :param filters: The generated filters according to the correct shape
        """
        return tf.nn.conv2d(input_layer, filters, strides=[1, 1, 1, 1], padding='SAME')
    
    def generate_max_pool_2by2(self, input_layer):
        """Generate a pooling layer on 2x2 by attaching the new input layer into it"""
        return tf.nn.max_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    def generate_convolutional_layer(self, shape, input_layer):
        """Generate a convolutional layer according to the shape (usually the shape of a CIFAR image) and the previous
            input layer
            :param shape: The defined shape, usually the shape of a CIFAR image (with width x heights x colours)
            :param input_layer: The previous input layer
        """
        filters = self.init_filters(shape)
        print("init filters with shape: " + str(shape))
        biases = self.init_biases([shape[3]])
        print("init biases with shape: " + str(shape[3]))
        return tf.nn.relu(self.generate_conv2d_layer(input_layer, filters) + biases)
    
    def generate_normal_full_layer(self, input_layer, size):
        """Generate classic layer with input flatten array and classical output
            :param input_layer: the previous layer from the convolution
            :param size: the previous size from the convolution
        """
        input_size = int(input_layer.get_shape()[1])
        filters = self.init_filters([input_size, size])
        biases = self.init_biases([size])
        return tf.matmul(input_layer, filters) + biases
    

def cnn_cifar():
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

    tfUtils = TFUtils()
    #Here the 4x4 represent the filter size while 3 32 represent pixels and colour of the image
    convo_1 = tfUtils.generate_convolutional_layer(input_layer=tfUtils.placeholder_training_batch, shape=[4, 4, 3, 32])
    convo_1_pooling = tfUtils.generate_max_pool_2by2(input_layer=convo_1)

    convo_2 = tfUtils.generate_convolutional_layer(input_layer=convo_1_pooling, shape=[4, 4, 32, 64])
    convo_2_pooling = tfUtils.generate_max_pool_2by2(input_layer=convo_2)

    #Here we flatten our previous layer to make it as input of our classical neural net
    convo_2_flatten = tf.reshape(convo_2_pooling, shape=[-1, 8*8*64])

    #The shape of 1024 generate an output of 1024 neurons
    full_layer_1 = tf.nn.relu(tfUtils.generate_normal_full_layer(input_layer=convo_2_flatten, size=1024))

    full_layer_dropout = tf.nn.dropout(full_layer_1, keep_prob=tfUtils.placeholder_dropout)

    prediction_layer = tfUtils.generate_normal_full_layer(input_layer=full_layer_dropout, size=10)
    print(prediction_layer)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tfUtils.placeholder_labels, logits=prediction_layer))
    print(cross_entropy)

    train_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = train_optimizer.minimize(cross_entropy)

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for i in range(500):
            batch = cfHelper.next_batch(batch_size=100)
            session.run(train, feed_dict={tfUtils.placeholder_training_batch: batch[0], tfUtils.placeholder_labels: batch[1], tfUtils.placeholder_dropout: 0.5})
            
            if i%100 == 0:
                matches = tf.equal(tf.argmax(prediction_layer, 1), tf.argmax(tfUtils.placeholder_labels, 1))
                accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
                session_run = session.run(accuracy, feed_dict={tfUtils.placeholder_training_batch: cfHelper.tests_images, tfUtils.placeholder_labels: cfHelper.tests_labels, tfUtils.placeholder_dropout: 1.0})
                print('Currently on step {}'.format(i))
                print('Accuracy is {}'.format(session_run))


cnn_cifar()
    