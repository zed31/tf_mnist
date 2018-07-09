import tensorflow as tf

def generate_conv2d_layer(placeholder, weighted_filter, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(input=placeholder, filter=weighted_filter, strides=strides, padding=padding)

def generate_maxpooling_2d(placeholder, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(value=placeholder, ksize=kernel_size, strides=strides, padding=padding)
