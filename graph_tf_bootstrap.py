#!/usr/bin/python3

import tensorflow as tf

"""
    Here we are going to see how graph is working
    when I talk about graph what I really mean is
    tensorflow execution graph
"""

"""
    Here we create a graph by making a default graph
"""
graph = tf.get_default_graph()

"""
    Here, we print out the list of operations the graph contains
    Operations: A list of expression that will be process when a session
    will run
"""
print(graph.get_operations())

constant_a = tf.constant(value=10, name='constant_a')

"""
    Here, we're going to have an operation that is basically a simple
    constant. When we create a constant we do create injecting a new
    operation inside the graph
"""
print(graph.get_operations())

"""
    We add a new constant inside the graph
"""
constant_b = tf.constant(value=20, name='constant_b')
print(graph.get_operations())

"""
    Result type won't be an Operation but a Tensor, however the graph
    will understand it as a lazy operation. However, the type ain't a
    constant but a Add type

    We can do the same for multiplication and the type would be a Mult
    we can also make some raw operations on Operation types by using
    the operators '+' '-' ... It will create Operation of correct
    type in the graph
"""
add_a_b = tf.add(x=constant_a, y=constant_b, name='add_a_b')
mult_a_b = tf.multiply(x=constant_a, y=constant_b, name='mult_a_b')
mult_add_a_b_mult_a_b = tf.multiply(x=add_a_b, y=mult_a_b, name='mult_add_a_b_mult_a_b')
print(graph.get_operations())

"""
    We can loop over all the graph operations and display their name like so:
"""
for operation in graph.get_operations(): print(operation.name)

"""
    We can run a session without initializing the global variables because
    we don't declare any variable in this piece of code
"""
with tf.Session() as sess:
    """
        When we execute a session that depends on other session, it will evaluate
        other sessions first then finish this piece of computation
    """
    result = sess.run(fetches=mult_add_a_b_mult_a_b)
    print(result)
