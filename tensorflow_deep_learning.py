#!/usr/bin/python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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
        self.NUM_NODE_HL1 = 500
        self.NUM_NODE_HL2 = 500
        self.NUM_NODE_HL3 = 500
        self.NUM_DATASET_CLASSES = 10
        self.BATCH_SIZE = 100

        self.placeholder_in = tf.placeholder(dtype=tf.float32, shape=[None, self.NUM_INPUT_NODE], name='input_placeholder')
        self.placeholder_out = tf.placeholder(dtype=tf.float32, name='output_placeholder')
    
    def setup_layers(self):
        """
            This function will setup the entire weights and biases of all the MNIST layers
            the tf.random_normal function will initialize a tensor with a shape of
            the random distribution
        """
        self.WEIGHTS_DATA = 'weights'
        self.BIASES_DATA = 'biases'

        self.hidden_layer_1 = {
            self.WEIGHTS_DATA: tf.Variable(tf.random_normal(shape=[self.NUM_INPUT_NODE, self.NUM_NODE_HL1])),
            self.BIASES_DATA: tf.Variable(tf.random_normal(shape=[self.NUM_NODE_HL1]))
        }

        self.hidden_layer_2 = {
            self.WEIGHTS_DATA: tf.Variable(tf.random_normal(shape=[self.NUM_NODE_HL1, self.NUM_NODE_HL2])),
            self.BIASES_DATA: tf.Variable(tf.random_normal(shape=[self.NUM_NODE_HL2]))
        }

        self.hidden_layer_3 = {
            self.WEIGHTS_DATA: tf.Variable(tf.random_normal(shape=[self.NUM_NODE_HL2, self.NUM_NODE_HL3])),
            self.BIASES_DATA: tf.Variable(tf.random_normal(shape=[self.NUM_NODE_HL3]))
        }

        self.output_layer = {
            self.WEIGHTS_DATA: tf.Variable(tf.random_normal(shape=[self.NUM_NODE_HL3, self.NUM_DATASET_CLASSES])),
            self.BIASES_DATA: tf.Variable(tf.random_normal(shape=[self.NUM_DATASET_CLASSES]))
        }

    def setup_computation_graph(self, data):
        """
            Setup all the Tensors operations needed by the graph
            to compute all activations of a simple node

            :param data: the data set used
        """
        balanced_sum_hl1 = tf.add(tf.matmul(data, self.hidden_layer_1[self.WEIGHTS_DATA]), self.hidden_layer_1[self.BIASES_DATA])
        self.activation_balanced_sum_hl1 = tf.nn.relu(balanced_sum_hl1)

        balanced_sum_hl2 = tf.add(tf.matmul(self.activation_balanced_sum_hl1, self.hidden_layer_2[self.WEIGHTS_DATA]), self.hidden_layer_2[self.BIASES_DATA])
        self.activation_balanced_sum_hl2 = tf.nn.relu(balanced_sum_hl2)

        balanced_sum_hl3 = tf.add(tf.matmul(self.activation_balanced_sum_hl2, self.hidden_layer_3[self.WEIGHTS_DATA]), self.hidden_layer_3[self.BIASES_DATA])
        self.activation_balanced_sum_hl3 = tf.nn.relu(balanced_sum_hl3)

        balanced_sum_output = tf.add(tf.matmul(self.activation_balanced_sum_hl3, self.output_layer[self.WEIGHTS_DATA]), self.output_layer[self.BIASES_DATA])
        self.activation_balance_sum_out = tf.nn.relu(balanced_sum_output)

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

    def train_operation(self):
        """
            This function is used to train the given tensorflow model
        """

engine = MNISTEngine()