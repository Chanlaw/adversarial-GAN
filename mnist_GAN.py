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
        gen_images = fully_connected(out2, 500, MNIST_SIZE, tf.nn.sigmoid, scope=scope)
        
#Discriminator network with 5 hidden layers and gaussian noise
real_images = tf.placeholder(tf.float32, [None, MNIST_SIZE], name="real_images")
labelled_images = tf.placeholder(tf.float32, [None, MNIST_SIZE], name="labelled_images")
with tf.variable_scope("discriminator"):
    with tf.variable_scope("hidden1") as scope:
        d1 = fully_connected(
                gaussian_noise(tf.concat([real_images, gen_images, labelled_images],0), sigma=0.3),
                MNIST_SIZE, 1000, train_scale=False, scope=scope)
    with tf.variable_scope("hidden2") as scope:
        d2 = fully_connected(gaussian_noise(d1, sigma=0.5), 1000, 500, train_scale=False, scope=scope)
    with tf.variable_scope("hidden3") as scope:
        d3 = fully_connected(gaussian_noise(d2, sigma=0.5), 500, 500, train_scale=False, scope=scope)
    with tf.variable_scope("hidden4") as scope:
        d4 = fully_connected(gaussian_noise(d3, sigma=0.5), 500, 250, train_scale=False, scope=scope)
    with tf.variable_scope("hidden5") as scope:
        d5 = fully_connected(gaussian_noise(d4, sigma=0.5), 250, 250, train_scale=False, scope=scope)
    with tf.variable_scope("output") as scope:
        # We only need 10 since we're fixing the logits of the "fake" category to be 0
        d_output = fully_connected(gaussian_noise(d5, sigma=0.5), 250, 10, None, scope=scope)

#Load MNIST Data
data = np.load('mnist.npz')
x_train = np.concatenate([data['x_train'], data['x_valid']], axis=0)
x_unl = x_train.copy()
x_unl2 = x_unl.copy()
y_train = np.concatenate([data['y_train'], data['y_valid']])
assert x_train.shape[0] == 60000
assert y_train.shape[0] == 60000
x_test = data['x_test']
y_test = data['y_test']
assert x_test.shape[0] == 10000
assert y_test.shape[0] == 10000

#Select labeled data
np.random.seed(2)
idx = np.random.permutation(x_train.shape[0]) 
x_train = x_train[idx]
y_train = y_train[idx]
x_labelled = []
y_labelled = []
for i in range(10):
    x_labelled.append(x_train[y_train==i][:FLAGS.examples_per_class])
    y_labelled.append(y_train[y_train==i][:FLAGS.examples_per_class])
x_labelled = np.concatenate(x_labelled, axis=0)
y_labelled = np.concatenate(y_labelled, axis=0)


#loss functions
#TODO: Implement label smoothing
num_unl = tf.placeholder(tf.int64, shape=[], name='num_unl')
num_lbl = tf.placeholder(tf.int64, shape=[], name='num_lbl') #num of labelled examples for supervised training
labels = tf.placeholder(tf.int64, shape=[None], name='labels')

with tf.name_scope('eval'):
    logits_unl, logits_fake, logits_lbl = tf.split(d_output, [num_unl, num_unl, num_lbl], 0)
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

    real_activations, fake_activations, _ = tf.split(d5, [num_unl, num_unl, num_lbl], 0) 
    loss_g = tf.reduce_mean(tf.square(tf.reduce_mean(real_activations,axis=0) 
                                    - tf.reduce_mean(fake_activations,axis=0))) 

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_lbl,axis=1),labels), tf.float32))
    prob_lbl = 1 - 1/(tf.reduce_sum(tf.exp(logits_lbl),axis=1)+1)
    over_half = tf.reduce_sum(tf.to_int32(tf.greater_equal(prob_lbl, 0.5)))
    avg_prob_lbl = tf.reduce_mean(prob_lbl)
    softmax_loss_lbl = tf.nn.softmax_cross_entropy_with_logits(logits=logits_lbl, 
                                                                labels=tf.one_hot(labels,10))
with tf.name_scope('adv'):
    #FGSM 
    epsilon=FLAGS.epsilon
    pertubation = tf.sign(tf.gradients(softmax_loss_lbl, labelled_images))
    #image summary for real_images, pertubation 
    #apply pertubation to images
    perturbed_images = tf.squeeze(epsilon * pertubation) + labelled_images

#create summary ops 
#merged summary for training: loss_g, loss_d, gen_images
with tf.name_scope('image_summaries'):
    gen_images_uint = tf.reshape(tf.cast((gen_images+epsilon)*255.0/(1+2*epsilon), tf.uint8), [batch_size, 28, 28, 1])
    images_uint = tf.reshape(tf.cast((labelled_images+epsilon)*255.0/(1+2*epsilon), tf.uint8), [batch_size, 28, 28, 1])
    pertubation_uint = tf.reshape(tf.cast(pertubation*127.5+127.5, tf.uint8),[batch_size, 28, 28, 1])

with tf.name_scope('loss'):
    loss_g_summary = tf.summary.scalar('generator', loss_g)
    loss_d_summary = tf.summary.scalar('discriminator', loss_d)
    loss_d_unl_summary = tf.summary.scalar('discriminator_unl', loss_d_unl)
    loss_d_fake_summary = tf.summary.scalar('discriminator_fake', loss_d_fake)
    loss_d_lbl_summary = tf.summary.scalar('discriminator_lbl', loss_d_lbl)
accuracy_summary = tf.summary.scalar('accuracy_d', accuracy)
gen_images_summary = tf.summary.image('gen_images', gen_images_uint, max_outputs=10)
train_summary = tf.summary.merge([loss_g_summary, loss_d_summary, loss_d_unl_summary, 
                                loss_d_fake_summary, loss_d_lbl_summary, accuracy_summary, 
                                gen_images_summary])
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

    for epoch in range(FLAGS.num_epochs):
        x_unl = x_unl[np.random.permutation(x_unl.shape[0])]
        x_unl2 = x_unl2[np.random.permutation(x_unl2.shape[0])]
        x_lbl = []
        y_lbl = []
        for i in range(int(math.ceil(x_unl.shape[0]/x_labelled.shape[0]))):
            idx = np.random.permutation(x_labelled.shape[0]) 
            x_lbl.append(x_labelled[idx])
            y_lbl.append(y_labelled[idx])
        x_lbl = np.concatenate(x_lbl, axis=0)
        y_lbl = np.concatenate(y_lbl, axis=0)
        print("-------")
        print("Epoch %d:" %(epoch + 1))
        print("-------")
        for i in range(int(x_train.shape[0]/batch_size)):
            feed_dict_d = {noise: np.random.randn(batch_size, 100), 
                        real_images: x_unl[i*batch_size: (i+1)*batch_size], 
                        labelled_images: x_lbl[i*batch_size: (i+1)*batch_size],
                        num_unl: batch_size,
                        num_lbl: len(y_lbl[i*batch_size: (i+1)*batch_size]),
                        labels: y_lbl[i*batch_size: (i+1)*batch_size]}
            feed_dict_g = {noise: np.random.randn(batch_size, 100), 
                        real_images: x_unl2[i*batch_size: (i+1)*batch_size], 
                        labelled_images: np.zeros([0,784]),
                        num_unl: batch_size,
                        num_lbl: 0,
                        labels: []}
            if ((i)%200 == 0):
                labelled_loss, unlabelled_loss, fake_loss, train_acc, disc_loss, _ , summary = \
                    sess.run([loss_d_lbl, loss_d_unl, loss_d_fake, accuracy, loss_d, d_train_op, 
                            train_summary], feed_dict=feed_dict_d)
                gen_loss, _ = sess.run([loss_g, g_train_op], feed_dict=feed_dict_g)
                print("Step %d, Discriminator Loss %.4f, Generator Loss %.4f" \
                        %((i + epoch*60000//batch_size), disc_loss, gen_loss))
                print("Discriminator Loss breakdown - Labelled %.4f, Fake %.4f, Unlabelled %.4f" \
                    %( labelled_loss, fake_loss, unlabelled_loss))
                train_writer.add_summary(summary, (i + epoch*60000//batch_size))
            else:
                _ = sess.run([d_train_op], feed_dict=feed_dict_d)
                _ = sess.run([g_train_op], feed_dict=feed_dict_g)
        #adversarial example evaluation
        test_probs=[]
        test_accuracies=[]
        adv_probs=[]
        adv_accuracies=[]
        detect_accuracies=[]
        print("- Evaluating on Test and Adverarial (epsilon=%.4f) Data:" %epsilon)
        eval_n = np.random.randint(int(x_test.shape[0]/batch_size))
        for i in range(int(x_test.shape[0]/batch_size)):
            feed_dict = {noise: np.zeros([0,100]),
                        real_images: np.zeros([0,784]),
                        labelled_images: x_test[i*batch_size: (i+1)*batch_size],
                        num_unl: 0,
                        num_lbl: batch_size,
                        labels: y_test[i*batch_size: (i+1)*batch_size]}
            if(i==eval_n):
                perturbed, disc_prob, disc_acc, disc_loss, num_correct, summary= \
                        sess.run([perturbed_images, avg_prob_lbl, accuracy, loss_d_lbl, over_half, test_summary], 
                        feed_dict=feed_dict)
                test_writer.add_summary(summary, epoch*60000//batch_size)
            else:
                perturbed, disc_prob, disc_acc, num_correct, disc_loss = \
                        sess.run([perturbed_images, avg_prob_lbl, accuracy, over_half, loss_d_lbl], 
                        feed_dict=feed_dict)
            perturbed_feed = {noise: np.zeros([0,100]),
                        real_images: np.zeros([0,784]),
                        labelled_images: perturbed,
                        num_unl: 0,
                        num_lbl: batch_size,
                        labels: y_test[i*batch_size: (i+1)*batch_size]}
            if(i == eval_n):
                perturbed_prob, perturbed_acc, perturbed_loss, num_incorrect, summary= \
                        sess.run([avg_prob_lbl, accuracy, loss_d_lbl, over_half, adv_summary], 
                                feed_dict=perturbed_feed)
                adv_writer.add_summary(summary, epoch*60000//batch_size)
            else:
                perturbed_prob, perturbed_acc, perturbed_loss, num_incorrect = \
                        sess.run([avg_prob_lbl, accuracy, loss_d_lbl, over_half], feed_dict=perturbed_feed)
            test_probs.append(disc_prob)
            test_accuracies.append(disc_acc)
            adv_probs.append(perturbed_prob)
            adv_accuracies.append(perturbed_acc)
            detect_accuracies.append(1.0*num_correct/batch_size)
            detect_accuracies.append(1.0 - 1.0*num_incorrect/batch_size)

        test_probs=np.array(test_probs)
        test_accuracies=np.array(test_accuracies)
        adv_probs=np.array(adv_probs)
        adv_accuracies=np.array(adv_accuracies)
        detect_accuracies = np.array(detect_accuracies)
        print("Original Accuracy: %.4f+/-%.4f Adversarial Accuracy: %.4f+/-%.4f" \
            %(np.mean(test_accuracies), np.std(test_accuracies), 
                np.mean(adv_accuracies), np.std(adv_accuracies)))
        print("Original Probability: %.4f+/-%.4f Adversarial Probability: %.4f+/-%.4f" \
            %(np.mean(test_probs), np.std(test_probs), 
                np.mean(adv_probs), np.std(adv_probs)))
        print("Adversarial Classification Accuracy  %.4f+/-%.4f" \
            %(np.mean(detect_accuracies), np.std(detect_accuracies)))

    
        #Make a checkpoint
        saver.save(sess, FLAGS.log_dir + '/checkpoint', global_step=(epoch+1))