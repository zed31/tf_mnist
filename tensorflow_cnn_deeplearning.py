#!/usr/bin/python3

import tensorflow as tf
import tf_utils
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 128
EPOCH_NUMBER = 10

class MNISTNeuralModel:

    """
        This class will initialize the entire tensorflow data model
        and will be used by the engine when a session is running
    """
    
    def __init__(self):
        """
            Here, we define some constants, the first is the number of neurons inside
            the first hidden layer, then the number of neurons on the second and finally on
            the third one.

            There also is the number of output per datasets (here 10) and the size of a
            training batch

            We also define our input node number which is 28x28 pixels: 784 pixels
        """
        self.NUM_INPUT_NODE = 784
        self.NUM_DATASET_CLASSES = 10
        self.DROPOUT_RATE = 0.8

        self.placeholder_in = tf.placeholder(
            dtype=tf.float32, 
            shape=[None, self.NUM_INPUT_NODE], 
            name='dataset_placeholder')
        self.placeholder_labels = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self.NUM_DATASET_CLASSES],
            name='labels_placeholder')
    
    def setup_layers(self):
        """
            This function will setup the entire weights and biases of all the MNIST layers
            the tf.random_normal function will initialize a tensor with a shape of
            the random distribution

            the first two elements of filter_convX value is the size of the square filter
            the second is the number of input and the last one is the number of output

            After applying the convolution and the pooling, we will enter on the fully connected
            layer with 7x7 input and with 64 input, there will be 7x7x64 total input
            it will output 1024 nodes
        """
        self.filters = {
            'filter_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            'filter_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            'fully_connected': tf.Variable(tf.random_normal([7*7*64, 1024])),
            'output': tf.Variable(tf.random_normal([1024, self.NUM_DATASET_CLASSES]))
        }

        self.biases = {
            'bias_conv1': tf.Variable(tf.random_normal([32])),
            'bias_conv2': tf.Variable(tf.random_normal([64])),
            'bias_fully_connected': tf.Variable(tf.random_normal([1024])),
            'bias_output': tf.Variable(tf.random_normal([self.NUM_DATASET_CLASSES]))
        }
        

    def setup_computation_graph(self, placeholder):
        """
            The goal of this function is to generate an output according
            to all the convolutional layers and fully connected layers
            we first reshape everything into a 4D tensor placeholder

            the shape will looks like 28x28 images with only one colour
            and the -1 indicate that the batch size will be evaluated later

            Once the last maxpooling is done we reshape the tensor into a 2x2
            tensor

            :param placeholder: the dataset placeholder used
        """
        placeholder = tf.reshape(tensor=placeholder, shape=[-1, 28, 28, 1])
        conv1 = tf.nn.relu(tf_utils.generate_conv2d_layer(
                    placeholder=placeholder, 
                    weighted_filter=self.filters['filter_conv1']) + self.biases['bias_conv1'])
        maxpool1 = tf_utils.generate_maxpooling_2d(placeholder=conv1)

        conv2 = tf.nn.relu(tf_utils.generate_conv2d_layer(
            placeholder=maxpool1,
            weighted_filter=self.filters['filter_conv2']) + self.biases['bias_conv2'])
        maxpool2 = tf.nn.relu(tf_utils.generate_maxpooling_2d(placeholder=conv2))

        fully_connected = tf.reshape(tensor=maxpool2, shape=[-1, 7*7*64])
        fully_connected = tf.nn.relu(tf.matmul(fully_connected, self.filters['fully_connected']) + self.biases['bias_fully_connected'])
        fully_connected = tf.nn.dropout(fully_connected, self.DROPOUT_RATE)
        output = tf.matmul(fully_connected, self.filters['output']) + self.biases['bias_output']
        return output
        
        
class MNISTEngine:

    """
        This class hold the engine and make all the computation for the NN
        by running the session
    """

    def __init__(self):
        """
            Initialize the mnist data set as a one hot vector and labelled data
        """
        self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        self.mnist_model = MNISTNeuralModel()
        self.mnist_model.setup_layers()

    def setup_train_operation(self):
        """
            This function is used to train the given tensorflow model
        """
        self.pred = self.mnist_model.setup_computation_graph(self.mnist_model.placeholder_in)
        self.loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels=self.mnist_model.placeholder_labels))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=self.loss)
        
    
    def run_epoch_session(self, epoch_size=EPOCH_NUMBER, batch_size=BATCH_SIZE):
        """
            Run a batch test using `epoch` epoch units
            :param epoch_size: The number of epoch training
            :param batch_size: The size of one training
        """
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epoch_size):
                epoch_loss = 0
                mnist_example_nbr = int(self.mnist.train.num_examples/batch_size)
                for _ in range(0, mnist_example_nbr):
                    input_batch, label_batch = self.mnist.train.next_batch(batch_size)
                    _, loss_range = sess.run(
                        fetches=[self.optimizer, self.loss], 
                        feed_dict={self.mnist_model.placeholder_in: input_batch, self.mnist_model.placeholder_labels: label_batch})
                    epoch_loss += loss_range
                print('Epoch', epoch, 'completed out of', epoch_size, 'loss', epoch_loss)
            correct = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.mnist_model.placeholder_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            accuracy_eval = accuracy.eval(feed_dict={self.mnist_model.placeholder_in: self.mnist.test.images, self.mnist_model.placeholder_labels: self.mnist.test.labels})
            print('Accuracy:', accuracy_eval)
        
engine = MNISTEngine()
engine.setup_train_operation()
engine.run_epoch_session()