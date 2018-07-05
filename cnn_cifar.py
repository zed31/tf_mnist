#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

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
    out = np.zeros(((n, vals)))
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
        print('Setting up the images and labels')

        self.training_images = np.vstack([d[b'data'] for d in self.all_train_batches])
        training_image_len = len(self.training_images)

        self.training_images = self.training_images.reshape(training_image_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.training_labels = one_hot_encoding(np.hstack([d[b'data'] for d in self.all_train_batches]), 10)

        print('Setting up test images and labels')
        self.tests_images = np.vstack([d[b'data'] for d in self.test_batch])
        test_images_length = len(self.tests_images)
        self.tests_images = self.tests_images.reshape(test_images_length, 3, 32, 32).transpose(0, 2, 3, 1) / 255

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

X = prepare_to_plot_images(data_batch1)
show_images(X[0:10])
