""" Replicating Odena et al's AC-GANs:
https://arxiv.org/abs/1610.09585v3
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
tf.app.flags.DEFINE_string('log_dir', 'checkpoints_ac', 
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
    with tf.variable_scope(scope, "fully_connected"):
        g = tf.get_variable("g", [num_units], initializer=tf.ones_initializer(), trainable=train_scale)
        V = tf.get_variable("weights", [input_len, num_units])
        W = g * tf.nn.l2_normalize(V, dim=0, epsilon=1e-12)
        b = tf.get_variable("biases", [num_units], initializer=tf.constant_initializer(0))
        outputs = tf.matmul(x, W) + b
        if activation is not None:
            outputs = activation(outputs)
        return outputs

#Generator network with 3 hidden layers with batch normalization
#TODO: Implement virtual batch norm 
labels_fake = tf.placeholder(tf.int64, shape=[None], name='fake_labels')
with tf.variable_scope("generator"):
    noise = tf.placeholder(tf.float32, [None, 100], name="noise")
    gen_input = tf.concat([tf.one_hot(labels_fake, 10), noise], 1)
    with tf.variable_scope("hidden1") as scope:
        out1 = tf.contrib.layers.batch_norm(
            tf.contrib.layers.fully_connected(gen_input, 500, tf.nn.softplus, scope=scope),
            scale=False, scope=scope)
    with tf.variable_scope("hidden2") as scope:
        out2 = tf.contrib.layers.batch_norm(
            tf.contrib.layers.fully_connected(out1, 500, tf.nn.softplus, scope=scope),
            scale=False, scope=scope)
    with tf.variable_scope("output") as scope:
        gen_images = fully_connected(out2, 500, MNIST_SIZE, tf.nn.sigmoid, scope=scope)
#Discriminator network with 5 hidden layers and gaussian noise
labels_real = tf.placeholder(tf.int64, shape=[None], name='real_labels')
real_images = tf.placeholder(tf.float32, [None, MNIST_SIZE], name="real_images")
with tf.variable_scope("discriminator"):
    with tf.variable_scope("hidden1") as scope:
        d1 = fully_connected(
                gaussian_noise(tf.concat([gen_images, real_images],0), sigma=0.3),
                MNIST_SIZE, 1000, train_scale=False, scope=scope)
    with tf.variable_scope("hidden2") as scope:
        d2 = fully_connected(gaussian_noise(d1, sigma=0.3), 1000, 500, train_scale=0, scope=scope)
    with tf.variable_scope("hidden3") as scope:
        d3 = fully_connected(gaussian_noise(d2, sigma=0.3), 500, 500, train_scale=0, scope=scope)
    with tf.variable_scope("hidden4") as scope:
        d4 = fully_connected(gaussian_noise(d3, sigma=0.3), 500, 500, train_scale=0, scope=scope)
    with tf.variable_scope("hidden5") as scope:
        d5 = fully_connected(gaussian_noise(d4, sigma=0.3), 500, 250, train_scale=0, scope=scope)
    with tf.variable_scope("output") as scope:
        # We only need 10 since we're fixing the logits of the "fake" category to be 0
        d_output = fully_connected(gaussian_noise(d5, sigma=0.3), 250, 10, None, scope=scope)

#Load MNIST Data
data = np.load('mnist.npz')
x_train = np.concatenate([data['x_train'], data['x_valid']], axis=0)
y_train = np.concatenate([data['y_train'], data['y_valid']])
assert x_train.shape[0] == 60000
assert y_train.shape[0] == 60000
x_test = data['x_test']
y_test = data['y_test']
assert x_test.shape[0] == 10000
assert y_test.shape[0] == 10000

#loss functions
#TODO: Implement label smoothing
num_unl = tf.placeholder(tf.int64, shape=[], name='num_unl')
num_real = tf.placeholder(tf.int64, shape=[], name='num_real') #num of labelled examples for supervised training

with tf.name_scope('eval'):
    logits_fake, logits_real = tf.split(d_output, [num_unl, num_real], 0)
    l_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_fake,
                                                            labels=tf.one_hot(labels_fake,10))) \
            + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_real, 
                                                            labels=tf.one_hot(labels_real,10)))
    l_source = tf.reduce_mean(- tf.log(tf.reduce_sum(tf.exp(logits_real), axis=1) + 1e-8) \
                + tf.log(tf.reduce_sum(tf.exp(logits_real),axis=1)+1)) \
                + tf.reduce_mean(tf.log(tf.reduce_sum(tf.exp(logits_fake),axis=1)+1))
    loss_d = l_source + l_class
    loss_g = -l_source + l_class

    # fake_activations, real_activations = tf.split_v(d5, [num_unl, num_real], 0) 
    # loss_g = tf.reduce_mean(tf.square(tf.reduce_mean(real_activations,axis=0) 
    #                                 - tf.reduce_mean(fake_activations,axis=0))) 

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_real,axis=1),labels_real), tf.float32))
    prob_lbl = 1 - 1/(tf.reduce_sum(tf.exp(logits_real),axis=1)+1)
    avg_prob_lbl = tf.reduce_mean(prob_lbl)
    softmax_loss_lbl = tf.nn.softmax_cross_entropy_with_logits(logits=logits_real, 
                                                                labels=tf.one_hot(labels_real,10))
with tf.name_scope('adv'):
    #FGSM 
    epsilon=FLAGS.epsilon
    pertubation = tf.sign(tf.gradients(softmax_loss_lbl, real_images))
    #image summary for real_images, pertubation 
    #apply pertubation to images
    perturbed_images = tf.squeeze(epsilon * pertubation) + real_images

#create summary ops 
#merged summary for training: loss_g, loss_d, gen_images
with tf.name_scope('image_summaries'):
    gen_images_uint = tf.reshape(tf.cast((gen_images+epsilon)*255.0/(1+2*epsilon), tf.uint8), [batch_size, 28, 28, 1])
    images_uint = tf.reshape(tf.cast((real_images+epsilon)*255.0/(1+2*epsilon), tf.uint8), [batch_size, 28, 28, 1])
    pertubation_uint = tf.reshape(tf.cast(pertubation*127.5+127.5, tf.uint8),[batch_size, 28, 28, 1])

with tf.name_scope('loss'):
    loss_g_summary = tf.summary.scalar('generator', loss_g)
    loss_d_summary = tf.summary.scalar('discriminator', loss_d)
    l_source_summary = tf.summary.scalar('source', l_source)
    l_class_summary = tf.summary.scalar('class', l_class)
accuracy_summary = tf.summary.scalar('accuracy_d', accuracy)
gen_images_summary = tf.summary.image('gen_images', gen_images_uint, max_outputs=10)
train_summary = tf.summary.merge([loss_g_summary, loss_d_summary, l_source_summary, 
                                l_class_summary, accuracy_summary, gen_images_summary])
#merged summary for evaluation and adversarial examples: accuracy, prob_lbl, images, pertubation, perturbed images
prob_lbl_mean_summary = tf.summary.scalar('prob_lbl_mean', avg_prob_lbl)
prob_lbl_summary = tf.summary.histogram('prob_lbl', prob_lbl)
image_summary = tf.summary.image('images', images_uint, max_outputs=10)
pertubation_summary = tf.summary.image('pertubation', pertubation_uint, max_outputs=10)
test_summary = tf.summary.merge([accuracy_summary, prob_lbl_mean_summary, prob_lbl_summary,
                                image_summary, pertubation_summary])
adv_summary = tf.summary.merge([accuracy_summary, prob_lbl_mean_summary, prob_lbl_summary,
                                image_summary])

with tf.name_scope('training'):
    d_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5)
    gvs = d_optimizer.compute_gradients(loss_d, 
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="discriminator"))
    clipped_gradients=[(tf.clip_by_norm(grad, FLAGS.max_gradient_norm), var) for grad, var in gvs]
    d_train_op = d_optimizer.apply_gradients(clipped_gradients)
    g_train_op = g_optimizer.minimize(loss_g,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="generator"))

if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

#train the network
saver = tf.train.Saver()
sess = tf.Session()
with sess.as_default():
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    adv_writer = tf.summary.FileWriter(FLAGS.log_dir + '/adv')
    init = tf.global_variables_initializer()
    sess.run(init)
    #load from checkpoint if necessary
    if (FLAGS.load_from_checkpoint):
        print("Loading from checkpoint: %s" %tf.train.latest_checkpoint(FLAGS.log_dir))
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))

    for epoch in xrange(FLAGS.num_epochs):
        idx = np.random.permutation(x_train.shape[0]) 
        x_train = x_train[idx]
        y_train = y_train[idx]
        print("-------")
        print("Epoch %d:" %(epoch + 1))
        print("-------")
        for i in xrange(int(x_train.shape[0]/batch_size)):
            feed_dict = {noise: np.random.randn(batch_size, 100), 
                        real_images: x_train[i*batch_size: (i+1)*batch_size], 
                        num_unl: batch_size,
                        num_real: batch_size,
                        labels_real: y_train[i*batch_size: (i+1)*batch_size],
                        labels_fake: np.concatenate((np.array([0,1,2,3,4,5,6,7,8,9]),
                                                    np.random.randint(10, size=batch_size-10)))}
            if ((i)%200 == 0):
                class_loss, source_loss, gen_loss, disc_loss, _, _, summary = \
                    sess.run([l_class, l_source, loss_g, loss_d, d_train_op, 
                            g_train_op, train_summary], feed_dict=feed_dict)
                print("Step %d, Discriminator Loss %.4f, Generator Loss %.4f" \
                        %((i + epoch*60000//batch_size), disc_loss, gen_loss))
                train_writer.add_summary(summary, (i + epoch*60000//batch_size))
            else:
                _, _ = sess.run([d_train_op, g_train_op], feed_dict=feed_dict)
        #adversarial example evaluation
        test_probs=[]
        test_accuracies=[]
        adv_probs=[]
        adv_accuracies=[]
        print("- Evaluating on Test and Adverarial (epsilon=%.4f) Data:" %epsilon)
        eval_n = np.random.randint(int(x_test.shape[0]/batch_size))
        for i in xrange(int(x_test.shape[0]/batch_size)):
            feed_dict = {noise: np.zeros([0,100]),
                        real_images: x_test[i*batch_size: (i+1)*batch_size],
                        num_unl: 0,
                        num_real: batch_size,
                        labels_real: y_test[i*batch_size: (i+1)*batch_size],
                        labels_fake: np.zeros(0)}
            if(i==eval_n):
                perturbed, disc_prob, disc_acc, summary= \
                        sess.run([perturbed_images, avg_prob_lbl, accuracy, test_summary], 
                        feed_dict=feed_dict)
                test_writer.add_summary(summary, (epoch+1)*60000//batch_size)
            else:
                perturbed, disc_prob, disc_acc = \
                        sess.run([perturbed_images, avg_prob_lbl, accuracy], 
                        feed_dict=feed_dict)
            perturbed_feed = {noise: np.zeros([0,100]),
                        real_images: perturbed,
                        num_unl: 0,
                        num_real: batch_size,
                        labels_real: y_test[i*batch_size: (i+1)*batch_size],
                        labels_fake:np.zeros(0)}
            if(i == eval_n):
                perturbed_prob, perturbed_acc, summary= \
                        sess.run([avg_prob_lbl, accuracy, adv_summary], 
                                feed_dict=perturbed_feed)
                adv_writer.add_summary(summary, (epoch+1)*60000//batch_size)
            else:
                perturbed_prob, perturbed_acc = \
                        sess.run([avg_prob_lbl, accuracy], feed_dict=perturbed_feed)
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