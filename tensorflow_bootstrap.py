#!/usr/bin/python3

"""
    This script is for training tensorflow data model
    there are no neural network on it, only computation using the tensor objects
"""

import tensorflow as tf

"""
    Here we create variable and constants
    the things you need to know when you're using tensorflow is that everything is tensor
    and everything is just expression that gets evaluated when you run the session
    so as everything is evaluated as tensor, you need to declare Variable and constants the
    same way you're declaring constant and variables on tensor.
"""
zero = tf.Variable(0)
one = tf.constant(1)

"""
    As tensors contains arithmetics data, it also contains arithmetics operations
    as well as assignations
"""
result = tf.add(zero, one)
update = tf.assign(zero, result)

"""
    We can also make some string operations
"""
hello = tf.constant("Hello")
world = tf.constant("World")

hello_world = tf.add(hello, world)

"""
    Here we create a placeholder, first advantage of the placeholder is that it
    can allows you to store more than one variable at a time but a group of
    variable instead

    Unlike constant or variable, placeholder doesn't store any value until a
    specific variable is given during the session. However you must specify the
    type first, the most of the case, it's a floating point
"""
placeholder_1 = tf.placeholder(tf.float32)

"""
    This placeholder has no value for now, it just know that for any value the
    placeholder_1 variable will have, the placeholder_2 will multiply it by 3
"""
placeholder_2 = placeholder_1*3

"""
    Here we make a call to the initializer which will initialize all the variables of the declared
    script, for instance here the zero variable will be initialized
"""
init_variables = tf.global_variables_initializer()

"""
    We create a session to run the graph
"""
with tf.Session() as session:
    """
        We need to run the global initializer to initialize the session properly
    """
    initialize_runner = session.run(fetches=init_variables)

    """
        Here we're going to run string operation
    """
    print(session.run(fetches=hello_world))

    """
        Here we run a session called "running session" which will evaluate the graph
        as a basic tensor expression
    """
    zero_session = session.run(fetches=zero)
    print(zero_session)

    """
        We run the placeholder, first thing you need to know is that
        the placeholder is, unlike variables, ran with a dictionnary
        mapping the placeholder variable and the value of the placeholder
    """
    placeholder_res = session.run(fetches=placeholder_2, feed_dict={placeholder_1: 3.5})
    print(placeholder_res)

    """
        You can do anything you want with that placeholder e.g: Feed with an array of value
        which will gives you a numpy array
    """
    placeholder_res_array = session.run(fetches=placeholder_2, feed_dict={placeholder_1: [1, 2, 3]})
    print(placeholder_res_array)

    """
        We can even make multi dimensionnal array
    """
    dictionnary = {placeholder_1: [[1, 2, 3], [1, 2, 3], [1, 2, 3]]}
    placeholder_res_multidim_array = session.run(fetches=placeholder_2, feed_dict=dictionnary)
    print(placeholder_res_multidim_array)

    for i in range(0, 5):
        """
            As everything is kind of lazy, we need to run a specific session to make sure
            the computation is done properly, for instance running the update will add
            1 to 0 and update the result inside the zero variable
        """
        update_session = session.run(fetches=update)
        """
            We then run the zero variable to get the result
        """
        print(session.run(fetches=zero))
