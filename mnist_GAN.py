""" Replicating feature-matching MNIST experiments in OpenAI's Improved GAN paper:
https://arxiv.org/abs/1606.03498
Ported to tensorflow from their improved-gan code:
https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_mnist_feature_matching.py
"""
from __future__ import absolute_import, division, print_function

import math

import numpy as np
import tensorflow as tf

#Define flags
tf.app.flags.DEFINE_integer('batch_size', 100,
                            "Number of digits to process in a batch.")
tf.app.flags.DEFINE_integer('num_epochs', 100,
                            "Number of times to process MNIST data.")
tf.app.flags.DEFINE_integer('examples_per_class', 100,
                            "Number of examples to give per MNIST Class")
tf.app.flags.DEFINE_integer('learning_rate', 0.0003,
                            "Learning rate for use in training")
FLAGS = tf.app.flags.FLAGS

#Define global variables
MNIST_SIZE = 784
batch_size=FLAGS.batch_size

def gaussian_noise(x, sigma):
    noise = tf.random_normal(tf.shape(x), 0, sigma)
    return x + noise 

def log_sum_exp(x): #to prevent overflow
  # max_x = tf.reduce_max(x, 1, keep_dims=True)
  # x -= max_x
  return tf.log(tf.reduce_sum(tf.exp(x), 1)) - tf.log(tf.reduce_sum(tf.exp(x))+1)

#Generator network with 3 hidden layers with batch normalization
#TODO: Implement virtual batch norm
with tf.variable_scope("generator"):
    noise = tf.placeholder(tf.float32, [None, 100], name="noise")
    with tf.variable_scope("hidden1") as scope:
        out1 = tf.contrib.layers.batch_norm(
            tf.contrib.layers.fully_connected(noise, 500, tf.nn.softplus, scope=scope),
            scope=scope)
    with tf.variable_scope("hidden2") as scope:
        out2 = tf.contrib.layers.batch_norm(
            tf.contrib.layers.fully_connected(out1, 500, tf.nn.softplus, scope=scope),
            scope=scope)
    with tf.variable_scope("output") as scope:
        gen_images = tf.contrib.layers.batch_norm(
            tf.contrib.layers.fully_connected(out2, MNIST_SIZE, tf.nn.sigmoid, scope=scope),
            scope=scope)
#Discriminator network with 5 hidden layers and gaussian noise
real_images = tf.placeholder(tf.float32, [None, MNIST_SIZE], name="real_images")
labelled_images = tf.placeholder(tf.float32, [None, MNIST_SIZE], name="labelled_images")
with tf.variable_scope("discriminator"):
    with tf.variable_scope("hidden1") as scope:
        d1 = tf.contrib.layers.fully_connected(
                gaussian_noise(tf.concat(0, [real_images, gen_images, labelled_images]), sigma=0.3),
                1000, scope=scope)
    with tf.variable_scope("hidden2") as scope:
        d2 = tf.contrib.layers.fully_connected(gaussian_noise(d1, sigma=0.5), 500, scope=scope)
    with tf.variable_scope("hidden3") as scope:
        d3 = tf.contrib.layers.fully_connected(gaussian_noise(d2, sigma=0.5), 250, scope=scope)
    with tf.variable_scope("hidden4") as scope:
        d4 = tf.contrib.layers.fully_connected(gaussian_noise(d3, sigma=0.5), 250, scope=scope)
    with tf.variable_scope("hidden5") as scope:
        d5 = tf.contrib.layers.fully_connected(gaussian_noise(d4, sigma=0.5), 250, scope=scope)
    with tf.variable_scope("output") as scope:
        # We only need 10 since we're fixing the logits of the "fake" category to be 0
        d_output = tf.contrib.layers.fully_connected(d5, 10, None, scope=scope)

#Load MNIST Data
data = np.load('mnist.npz')
x_train = np.concatenate([data['x_train'], data['x_valid']], axis=0)
x_unl = x_train.copy()
y_train = np.concatenate([data['y_train'], data['y_valid']])
assert x_train.shape[0] == 60000
assert y_train.shape[0] == 60000
x_test = data['x_test']
y_test = data['y_test']
assert x_test.shape[0] == 10000
assert y_test.shape[0] == 10000

#Select labeled data
idx = np.random.permutation(x_train.shape[0]) 
x_train = x_train[idx]
y_train = y_train[idx]
x_labelled = []
y_labelled = []
for i in xrange(10):
    x_labelled.append(x_train[y_train==i][:FLAGS.examples_per_class])
    y_labelled.append(y_train[y_train==i][:FLAGS.examples_per_class])
x_labelled = np.concatenate(x_labelled, axis=0)
y_labelled = np.concatenate(y_labelled, axis=0)

#loss functions
#TODO: Implement label smoothing
num_unl = tf.placeholder(tf.int64, shape=[])
num_lbl = tf.placeholder(tf.int64, shape=[]) #num of labelled examples for supervised training
labels = tf.placeholder(tf.int64, shape=[None])

logits_unl, logits_fake, logits_lbl = tf.split(d_output, [num_unl, num_unl, num_lbl],0)

loss_d_unl = tf.reduce_mean((- tf.log(tf.reduce_sum(tf.exp(logits_unl), 1)) + tf.log(tf.reduce_sum(tf.exp(logits_unl))+1)))
loss_d_lbl = tf.cond(tf.greater(num_lbl, 0), 
                    lambda: tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(logits_lbl, tf.one_hot(labels, 10))), 
                    lambda: tf.constant(0.))
loss_d_fake = tf.reduce_mean(tf.nn.softplus(logits_fake))
loss_d = loss_d_unl + loss_d_lbl + loss_d_fake

real_activations, fake_activations, _ = tf.split(d5, [num_unl, num_unl, num_lbl], 0) 
loss_g = tf.reduce_mean(tf.square(tf.reduce_mean(real_activations,axis=0) 
                                - tf.reduce_mean(fake_activations,axis=0))) 
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_lbl,axis=1),labels), tf.float32))

#train the network
sess = tf.Session()
with sess.as_default():
    d_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    gvs = d_optimizer.compute_gradients(loss_d, 
                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="discriminator"))
    clipped_gradients=[(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gvs]
    d_train_op = d_optimizer.apply_gradients(clipped_gradients)
    g_train_op = g_optimizer.minimize(loss_g,
                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="generator"))
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in xrange(FLAGS.num_epochs):
        print("-------")
        print("Epoch %d:" %(epoch+1))
        print("-------")
        for i in xrange(int(x_train.shape[0]/batch_size)):
            if y_labelled[i*batch_size: (i+1)*batch_size].size > 0:
                feed_dict = {noise: np.random.randn(batch_size, 100), 
                            real_images: x_unl[i*batch_size: (i+1)*batch_size], 
                            labelled_images: x_labelled[i*batch_size: (i+1)*batch_size],
                            num_unl: batch_size,
                            num_lbl: len(y_labelled[i*batch_size: (i+1)*batch_size]),
                            labels: y_labelled[i*batch_size: (i+1)*batch_size]}
                fake_loss, fake_images, train_acc, disc_loss, gen_loss, _ , _ = \
                        sess.run([loss_d_fake, out1, accuracy, loss_d, loss_g, d_train_op, g_train_op], 
                                feed_dict=feed_dict)
            else:
                feed_dict = {noise: np.random.randn(batch_size, 100), 
                            real_images: x_unl[i*batch_size: (i+1)*batch_size], 
                            labelled_images: np.zeros((0,784)),
                            num_unl: batch_size,
                            num_lbl: 0,
                            labels: []}
                unlabelled_logits, unlabelled_loss, labelled_loss, fake_loss, fake_images, disc_loss, gen_loss, _ , _ = sess.run([logits_unl, loss_d_unl, loss_d_lbl,
                    loss_d_fake, out1, loss_d, loss_g, d_train_op, g_train_op], 
                                                    feed_dict=feed_dict)

            if ((i+1)%200 == 0):
                print("Step %d, Discriminator Loss %.4f, Fake Loss %.4f, Generator Loss %.4f" \
                        %((i+1 + epoch*60000/batch_size), disc_loss,fake_loss, gen_loss))
                print("Loss breakdown - Labelled Loss %.4f, Fake Loss %.4f, Unlabelled Loss %.4f" %( labelled_loss, fake_loss, unlabelled_loss))
        test_losses=[]
        test_accuracies=[]
        for i in xrange(int(x_test.shape[0]/batch_size)):
            feed_dict = {noise: np.zeros([0,100]),
                        real_images: np.zeros([0,784]),
                        labelled_images: x_test[i*batch_size: (i+1)*batch_size],
                        num_unl: 0,
                        num_lbl: batch_size,
                        labels: y_test[i*batch_size: (i+1)*batch_size]}
            disc_acc, disc_loss = sess.run([accuracy, loss_d_lbl], feed_dict=feed_dict)
            test_losses.append(disc_loss)
            test_accuracies.append(disc_acc)
        print("Test Loss: %.4f Test Accuracy: %.4f" \
                %(sum(test_losses)/len(test_losses), sum(test_accuracies)/len(test_accuracies)))
