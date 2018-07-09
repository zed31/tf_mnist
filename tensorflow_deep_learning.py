#!/usr/bin/python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
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
        self.NUM_NODE_HL1 = 500
        self.NUM_NODE_HL2 = 500
        self.NUM_NODE_HL3 = 500
        self.NUM_DATASET_CLASSES = 10

        self.placeholder_in = tf.placeholder(dtype=tf.float32, shape=[None, self.NUM_INPUT_NODE], name='dataset_placeholder')
        self.placeholder_labels = tf.placeholder(dtype=tf.float32, name='labels_placeholder')
    
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

            :param data: the dataset placeholder used
        """
        balanced_sum_hl1 = tf.add(
            x=tf.matmul(data, self.hidden_layer_1[self.WEIGHTS_DATA], name='hl1_weighted_input'), 
            y=self.hidden_layer_1[self.BIASES_DATA], 
            name='hl1_balanced_sum')
        self.activation_balanced_sum_hl1 = tf.nn.relu(balanced_sum_hl1)

        balanced_sum_hl2 = tf.add(
            x=tf.matmul(self.activation_balanced_sum_hl1, self.hidden_layer_2[self.WEIGHTS_DATA], name='hl2_weighted_input'), 
            y=self.hidden_layer_2[self.BIASES_DATA], 
            name='hl2_balanced_sum')
        self.activation_balanced_sum_hl2 = tf.nn.relu(balanced_sum_hl2)

        balanced_sum_hl3 = tf.add(
            x=tf.matmul(self.activation_balanced_sum_hl2, self.hidden_layer_3[self.WEIGHTS_DATA]), 
            y=self.hidden_layer_3[self.BIASES_DATA],
            name='hl3_balanced_sum')
        self.activation_balanced_sum_hl3 = tf.nn.relu(balanced_sum_hl3)

        balanced_sum_output = tf.add(
            x=tf.matmul(self.activation_balanced_sum_hl3, self.output_layer[self.WEIGHTS_DATA]), 
            y=self.output_layer[self.BIASES_DATA],
            name='out_balanced_sum')

        return balanced_sum_output

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
        self.prediction_output = self.mnist_model.setup_computation_graph(data=self.mnist_model.placeholder_in)
        self.loss = tf.reduce_mean(
            input_tensor=tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.prediction_output, 
                labels=self.mnist_model.placeholder_labels))
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
                    _, c = sess.run(
                            fetches=[self.optimizer, self.loss], 
                            feed_dict={self.mnist_model.placeholder_in: input_batch, self.mnist_model.placeholder_labels: label_batch})
                    epoch_loss += c
                print('Epoch', epoch, 'completed out of', epoch_size, 'loss', epoch_loss)
            correct = tf.equal(tf.argmax(self.prediction_output, 1), tf.argmax(self.mnist_model.placeholder_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            print('Accuracy:', accuracy.eval({self.mnist_model.placeholder_in: self.mnist.test.images, self.mnist_model.placeholder_labels: self.mnist.test.labels}))


engine = MNISTEngine()
engine.setup_train_operation()
engine.run_epoch_session()