# Generator network
import nn_helper
import tensorflow as tf

MNIST_DIMENSION = 784

def build_generator():
    """Build graph for the generator network with 2 hidden layers with batch normalization.
    Returns:
        noise: Placeholder for noise input into the generator
        gen_images: generated images as output
    TODO:
        Implement virtual batch norm
    """
    with tf.variable_scope("generator"):
        noise = tf.placeholder(tf.float32, [None, 100], name="noise")
        with tf.variable_scope("hidden1") as scope:
            out1 = tf.contrib.layers.batch_norm(
                tf.contrib.layers.fully_connected(noise, 500, tf.nn.softplus, scope=scope),
                scale=False, scope=scope)
        with tf.variable_scope("hidden2") as scope:
            out2 = tf.contrib.layers.batch_norm(
                tf.contrib.layers.fully_connected(out1, 500, tf.nn.softplus, scope=scope),
                scale=False, scope=scope)
        with tf.variable_scope("output") as scope:
            gen_images = nn_helper.fully_connected(out2, 500, MNIST_DIMENSION, tf.nn.sigmoid, scope=scope)
        return noise, gen_images

def calculate_loss(d5, num_unl, num_lbl):
    """Calculate the loss from real and fake activations in generator using feature matching objective
    Args:
        d5: output from the last hidden layer in discriminator
        num_unl: number of generated and real unlabelled data
        num_lbl: number of labelled data
    """
    real_activations, fake_activations, _ = tf.split(d5, [num_unl, num_unl, num_lbl], 0) 
    loss_g = tf.reduce_mean(tf.square(tf.reduce_mean(real_activations,axis=0) 
                                    - tf.reduce_mean(fake_activations,axis=0))) 
    return loss_g

def train(loss_g, learning_rate):
    """Build the train op for generator
    Args:
        loss_g: loss function for generator from calculate_loss
        learning_rate: learning rate for generator
    Returns:
        train op for generator
    """
    g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
    g_train_op = g_optimizer.minimize(loss_g,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="generator"))
    return g_train_op
