import tensorflow as tf

def generate_conv2d_layer(placeholder, weighted_filter, strides=[1, 1, 1, 1], padding='SAME'):
    """
        This function generate a convolutional 2d layer

        :param placeholder: The placeholder used to link to
        :param weighted_filter: The weights / filters defined
        :param strides: The strides of the layer
        :param padding: The defined padding of the layer
    """
    return tf.nn.conv2d(input=placeholder, filter=weighted_filter, strides=strides, padding=padding)

def generate_maxpooling_2d(placeholder, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    """
        This function generates a pooling 2d layer

        :param placeholder: The placeholder used to link to
        :param kernel_size: The matrix used to map the content
        :param strides: The strides to jump to
        :param padding: The padding of the pooling layer
    """
    return tf.nn.max_pool(value=placeholder, ksize=kernel_size, strides=strides, padding=padding)
