""" Replicating feature-matching MNIST experiments in OpenAI's Improved GAN paper:
https://arxiv.org/abs/1606.03498
Ported to tensorflow from their improved-gan code:
https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_mnist_feature_matching.py
"""
from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import tensorflow as tf

#Define flags
tf.app.flags.DEFINE_integer('batch_size', 100,
                            "Number of digits to process in a batch.")
tf.app.flags.DEFINE_integer('num_epochs', 300,
                            "Number of times to process MNIST data.")
tf.app.flags.DEFINE_integer('examples_per_class', 100,
                            "Number of examples to give per MNIST Class.")
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                            "Learning rate for use in training.")
tf.app.flags.DEFINE_float('max_gradient_norm', 100.0,
                            "Clip discriminator gradients to this norm.")
tf.app.flags.DEFINE_string('log_dir', 'checkpoints', 
                            "Directory to put checkpoints and tensorboard log files.")
tf.app.flags.DEFINE_boolean('load_from_checkpoint', False,
                            "Whether or not we should load a pretrained model from checkpoint.")
tf.app.flags.DEFINE_float('epsilon', 0.25,
                            "Size of adversarial pertubation.")
FLAGS = tf.app.flags.FLAGS

#Define global variables
MNIST_SIZE = 784
batch_size=FLAGS.batch_size

def gaussian_noise(x, sigma):
    noise = tf.random_normal(tf.shape(x), 0, sigma)
    return x + noise 

def fully_connected(x, input_len, num_units, activation=tf.nn.relu, train_scale=True, scope=None):
    #simple fully connected layer with weight scaling
    with tf.variable_scope("fully_connected"):
        g = tf.get_variable("g", [num_units], initializer=tf.ones_initializer(), trainable=train_scale)
        V = tf.get_variable("weights", [input_len, num_units])
        W = g * V / (1e-8 + tf.sqrt(tf.reduce_sum(tf.square(V),axis=0)))
        b = tf.get_variable("biases", [num_units], initializer=tf.constant_initializer(0))
        outputs = tf.matmul(x, W) + b
        if activation is not None:
            outputs = activation(outputs)
        return outputs

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
        d1 = fully_connected(
                gaussian_noise(tf.concat_v2([real_images, gen_images, labelled_images],0), sigma=0.3),
                MNIST_SIZE, 1000, train_scale=False, scope=scope)
    with tf.variable_scope("hidden2") as scope:
        d2 = fully_connected(gaussian_noise(d1, sigma=0.5), 1000, 500, train_scale=False, scope=scope)
    with tf.variable_scope("hidden3") as scope:
        d3 = fully_connected(gaussian_noise(d2, sigma=0.5), 500, 250, train_scale=False, scope=scope)
    with tf.variable_scope("hidden4") as scope:
        d4 = fully_connected(gaussian_noise(d3, sigma=0.5), 250, 250, train_scale=False, scope=scope)
    with tf.variable_scope("hidden5") as scope:
        d5 = fully_connected(gaussian_noise(d4, sigma=0.5), 250, 250, train_scale=False, scope=scope)
    with tf.variable_scope("output") as scope:
        # We only need 10 since we're fixing the logits of the "fake" category to be 0
        d_output = fully_connected(gaussian_noise(d5, sigma=0.5), 250, 10, None, scope=scope)

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
np.random.seed(0)
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

logits_unl, logits_fake, logits_lbl = tf.split_v(d_output, [num_unl, num_unl, num_lbl], 0)
#TODO: Historical Averaging
loss_d_unl = 0.5 * tf.reduce_mean(- tf.log(tf.reduce_sum(tf.exp(logits_unl), axis=1) + 1e-8) \
            + tf.log(tf.reduce_sum(tf.exp(logits_unl),axis=1)+1))
loss_d_fake = 0.5 * tf.reduce_mean(tf.log(tf.reduce_sum(tf.exp(logits_fake),axis=1)+1))
loss_d_lbl = tf.cond(tf.greater(num_lbl, 0), 
                    lambda: tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(logits=logits_lbl, 
                                                                labels=tf.one_hot(labels,10))),
                    lambda: tf.constant(0.))
loss_d = loss_d_unl + loss_d_fake + loss_d_lbl 

real_activations, fake_activations, _ = tf.split_v(d5, [num_unl, num_unl, num_lbl], 0) 
loss_g = tf.reduce_mean(tf.square(tf.reduce_mean(real_activations,axis=0) 
                                - tf.reduce_mean(fake_activations,axis=0))) 

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_lbl,axis=1),labels), tf.float32))
prob_lbl = 1 - tf.reduce_mean(1/(tf.reduce_sum(tf.exp(logits_lbl),axis=1)+1))

softmax_loss_lbl = tf.nn.softmax_cross_entropy_with_logits(logits=logits_lbl, 
                                                            labels=tf.one_hot(labels,10))

#FGSM 
epsilon=FLAGS.epsilon
pertubation = epsilon * tf.sign(tf.gradients(softmax_loss_lbl, labelled_images))
#image summary for real_images, pertubation 
#apply pertubation to images
perturbed_images = tf.squeeze(pertubation) + labelled_images

#create summary ops 
#merged summary for training: loss_g, loss_d, gen_images
#merged summary for evaluation: accuracy, prob_lbl, loss_d, real_images
#merged summary for adversarial examples: accuracy, prob_lbl, loss_d, real_images, pertubation

if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

#train the network
saver = tf.train.Saver()
sess = tf.Session()
with sess.as_default():
    d_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5)
    gvs = d_optimizer.compute_gradients(loss_d, 
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="discriminator"))
    clipped_gradients=[(tf.clip_by_norm(grad, FLAGS.max_gradient_norm), var) for grad, var in gvs]
    d_train_op = d_optimizer.apply_gradients(clipped_gradients)
    g_train_op = g_optimizer.minimize(loss_g,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="generator"))
    init = tf.global_variables_initializer()
    sess.run(init)
    #load from checkpoint if necessary
    if (FLAGS.load_from_checkpoint):
        print("Loading from checkpoint: %s" %tf.train.latest_checkpoint(FLAGS.log_dir))
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))

    for epoch in xrange(FLAGS.num_epochs):
        idx = np.random.permutation(x_unl.shape[0]) 
        x_unl = x_unl[idx]
        x_lbl = []
        y_lbl = []
        for i in xrange(int(math.ceil(x_unl.shape[0]/x_labelled.shape[0]))):
            idx = np.random.permutation(x_labelled.shape[0]) 
            x_lbl.append(x_labelled[idx])
            y_lbl.append(y_labelled[idx])
        x_lbl = np.concatenate(x_lbl, axis=0)
        y_lbl = np.concatenate(y_lbl, axis=0)
        print("-------")
        print("Epoch %d:" %(epoch + 1))
        print("-------")
        for i in xrange(int(x_train.shape[0]/batch_size)):
            feed_dict = {noise: np.random.randn(batch_size, 100), 
                        real_images: x_unl[i*batch_size: (i+1)*batch_size], 
                        labelled_images: x_lbl[i*batch_size: (i+1)*batch_size],
                        num_unl: batch_size,
                        num_lbl: len(y_lbl[i*batch_size: (i+1)*batch_size]),
                        labels: y_lbl[i*batch_size: (i+1)*batch_size]}
            labelled_loss, unlabelled_loss, fake_loss, train_acc, disc_loss, gen_loss, _ , _ = \
                    sess.run([loss_d_lbl, loss_d_unl, loss_d_fake, accuracy, loss_d, loss_g,
                            d_train_op, g_train_op], 
                            feed_dict=feed_dict)
            if ((i)%200 == 0):
                print("Step %d, Discriminator Loss %.4f, Generator Loss %.4f" \
                        %((i + epoch*60000/batch_size), disc_loss, gen_loss))
                print("Discriminator Loss breakdown - Labelled %.4f, Fake %.4f, Unlabelled %.4f" \
                    %( labelled_loss, fake_loss, unlabelled_loss))
        #adversarial example evaluation
        test_probs=[]
        test_accuracies=[]
        adv_probs=[]
        adv_accuracies=[]
        print("- Evaluating on Test and Adverarial (epsilon=%.4f) Data:" %epsilon)
        for i in xrange(int(x_test.shape[0]/batch_size)):
            feed_dict = {noise: np.zeros([0,100]),
                        real_images: np.zeros([0,784]),
                        labelled_images: x_test[i*batch_size: (i+1)*batch_size],
                        num_unl: 0,
                        num_lbl: batch_size,
                        labels: y_test[i*batch_size: (i+1)*batch_size]}
            perturbed, disc_prob, disc_acc, disc_loss = \
                        sess.run([perturbed_images, prob_lbl, accuracy, loss_d_lbl], feed_dict=feed_dict)
            perturbed_feed = {noise: np.zeros([0,100]),
                        real_images: np.zeros([0,784]),
                        labelled_images: perturbed,
                        num_unl: 0,
                        num_lbl: batch_size,
                        labels: y_test[i*batch_size: (i+1)*batch_size]}
            perturbed_prob, perturbed_acc, perturbed_loss = \
                        sess.run([prob_lbl, accuracy, loss_d_lbl], feed_dict=perturbed_feed)
            test_probs.append(disc_prob)
            test_accuracies.append(disc_acc)
            adv_probs.append(perturbed_prob)
            adv_accuracies.append(perturbed_acc)

        test_probs=np.array(test_probs)
        test_accuracies=np.array(test_accuracies)
        adv_probs=np.array(adv_probs)
        adv_accuracies=np.array(adv_accuracies)
        print("Original Accuracy: %.4f+/-%.4f Adversarial Accuracy: %.4f+/-%.4f" \
            %(np.mean(test_accuracies), np.std(test_accuracies), 
                np.mean(adv_accuracies), np.std(adv_accuracies)))
        print("Original Probability: %.4f+/-%.4f Adversarial Probability: %.4f+/-%.4f" \
            %(np.mean(test_probs), np.std(test_probs), 
                np.mean(adv_probs), np.std(adv_probs)))
    
        #Make a checkpoint
        saver.save(sess, FLAGS.log_dir + '/checkpoint', global_step=(epoch+1))