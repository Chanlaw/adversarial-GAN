# Helper functions for creating neural nets

import tensorflow as tf

def fully_connected(x, input_len, num_units, activation=tf.nn.relu, train_scale=True, scope=None):
    """simple fully connected layer with weight scaling, used for discriminator
    Args:
        x: input to the layer
        input_len: number of dimensions for x
        num_units: number of units for this layer
        activation: tf activation function used for this layer
        train_scale: boolean indicates if variable g is set to trainable
        scope: variable_scope name, default name being fully_connected
    Returns:
        output for the layer
    """
    with tf.variable_scope(scope, "fully_connected"):
        g = tf.get_variable("g", [num_units], initializer=tf.ones_initializer(), trainable=train_scale)
        V = tf.get_variable("weights", [input_len, num_units])
        W = g * tf.nn.l2_normalize(V, dim=0, epsilon=1e-12)
        b = tf.get_variable("biases", [num_units], initializer=tf.constant_initializer(0))
        outputs = tf.matmul(x, W) + b
        if activation is not None:
            outputs = activation(outputs)
        return outputs

def gaussian_noise(x, sigma):
    """ Apply gaussian_noise to an input x with mean 0 and standard deviation sigma
    Args:
        x: original input
        sigma: standard deviation for the noise
    Returns:
        input x with noise
    """
    noise = tf.random_normal(tf.shape(x), 0, sigma)
    return x + noise 