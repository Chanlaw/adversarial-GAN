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

import adversarial
import discriminator
import generator
import load_and_preprocess
import nn_helper

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
MNIST_DIMENSION = 784
batch_size = FLAGS.batch_size

#Create folder for logging if it doesn't exist yet
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

#TODO: Implement label smoothing
#Define tf placeholders
unlabelled_images = tf.placeholder(tf.float32, [None, MNIST_DIMENSION], name="unlabelled_images")
labelled_images = tf.placeholder(tf.float32, [None, MNIST_DIMENSION], name="labelled_images")
num_unl = tf.placeholder(tf.int64, shape=[], name='num_unl')
num_lbl = tf.placeholder(tf.int64, shape=[], name='num_lbl') #num of labelled examples for supervised training
labels = tf.placeholder(tf.int64, shape=[None], name='labels')
noise = tf.placeholder(tf.float32, [None, 100], name="noise")

#Build generator and discriminator graph
gen = generator.MNIST_generator(noise)
dcm = discriminator.MNIST_discriminator(gen.fake_images, unlabelled_images, labelled_images, num_unl, num_lbl)

#define loss functions, accuracy and predictions ops
with tf.name_scope('eval'):
    dcm.calculate_loss(num_lbl, labels)
    gen.calculate_loss(dcm.d5, num_unl, num_lbl)

    #Discriminator accuracy for labeled data
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(dcm.logits_lbl,axis=1),labels), tf.float32))

    #Probabiliy of discriminator thinking that the labelled data are real
    #They should all be real
    prob_lbl = 1 - 1/(tf.reduce_sum(tf.exp(dcm.logits_lbl),axis=1)+1)

    #Counting >0.5 probabilty as real, the number of labelled images the discriminator thinks are real
    over_half = tf.reduce_sum(tf.to_int32(tf.greater_equal(prob_lbl, 0.5)))

    #Average probability of discriminator assigning labelled data real, accurate if close to 1
    avg_prob_lbl = tf.reduce_mean(prob_lbl)

with tf.name_scope('training'):
    d_train_op = dcm.train(dcm.loss_d, FLAGS.max_gradient_norm, FLAGS.learning_rate)
    g_train_op = gen.train(gen.loss_g, FLAGS.learning_rate)

#Perform FGSM attack to labelled_images
softmax_loss_lbl = tf.nn.softmax_cross_entropy_with_logits(logits=dcm.logits_lbl, 
                                                            labels=tf.one_hot(labels,10))
pertubation, perturbed_images = adversarial.fgsm_attack(labelled_images, softmax_loss_lbl, FLAGS.epsilon)

#create summary ops 
#merged summary for training: loss_g, loss_d, fake_images
with tf.name_scope('image_summaries'):
    fake_images_unit = tf.reshape(tf.cast((gen.fake_images+FLAGS.epsilon)*255.0/(1+2*FLAGS.epsilon), tf.uint8), [batch_size, 28, 28, 1])
    images_uint = tf.reshape(tf.cast((labelled_images+FLAGS.epsilon)*255.0/(1+2*FLAGS.epsilon), tf.uint8), [batch_size, 28, 28, 1])
    pertubation_uint = tf.reshape(tf.cast(pertubation*127.5+127.5, tf.uint8),[batch_size, 28, 28, 1])

with tf.name_scope('loss'):
    loss_g_summary = tf.summary.scalar('generator', gen.loss_g)
    loss_d_summary = tf.summary.scalar('discriminator', dcm.loss_d)
    loss_d_unl_summary = tf.summary.scalar('discriminator_unl', dcm.loss_d_unl)
    loss_d_fake_summary = tf.summary.scalar('discriminator_fake', dcm.loss_d_fake)
    loss_d_lbl_summary = tf.summary.scalar('discriminator_lbl', dcm.loss_d_lbl)


accuracy_summary = tf.summary.scalar('accuracy_d', accuracy)
fake_images_summary = tf.summary.image('fake_images', fake_images_unit, max_outputs=10)
train_summary = tf.summary.merge([loss_g_summary, loss_d_summary, loss_d_unl_summary, 
                                loss_d_fake_summary, loss_d_lbl_summary, accuracy_summary, 
                                fake_images_summary])

#merged summary for evaluation and adversarial examples: accuracy, prob_lbl, images, pertubation, perturbed images
prob_lbl_mean_summary = tf.summary.scalar('prob_lbl_mean', avg_prob_lbl)
prob_lbl_summary = tf.summary.histogram('prob_lbl', prob_lbl)
image_summary = tf.summary.image('images', images_uint, max_outputs=10)
pertubation_summary = tf.summary.image('pertubation', pertubation_uint, max_outputs=10)
test_summary = tf.summary.merge([accuracy_summary, prob_lbl_mean_summary, prob_lbl_summary,
                                image_summary, pertubation_summary])
adv_summary = tf.summary.merge([accuracy_summary, prob_lbl_mean_summary, prob_lbl_summary,
                                image_summary])

(x_train, y_train), (x_test, y_test) = load_and_preprocess.load_mnist_data()
x_labelled, y_labelled = load_and_preprocess.select_labelled_data(FLAGS.examples_per_class)
x_unl = x_train.copy()
x_unl2 = x_train.copy()

#Training the GAN
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
        x_unl = x_unl[np.random.permutation(x_unl.shape[0])] #Copy of x_train in random order
        x_unl2 = x_unl2[np.random.permutation(x_unl2.shape[0])] #Copy of x_train in random order
        x_lbl = []
        y_lbl = []

        # Repeat the labelled data so that there are similar amount of labelled data as unlabelled data
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
            batch_x_unl = x_unl[i*batch_size: (i+1)*batch_size]
            feed_dict_d = {noise: np.random.randn(batch_size, 100), 
                        unlabelled_images: batch_x_unl, 
                        labelled_images: x_lbl[i*batch_size: (i+1)*batch_size],
                        num_unl: len(batch_x_unl),
                        num_lbl: len(y_lbl[i*batch_size: (i+1)*batch_size]),
                        labels: y_lbl[i*batch_size: (i+1)*batch_size]}
            feed_dict_g = {noise: np.random.randn(batch_size, 100), 
                        unlabelled_images: x_unl2[i*batch_size: (i+1)*batch_size], 
                        labelled_images: np.zeros([0,784]),
                        num_unl: batch_size,
                        num_lbl: 0,
                        labels: []}

            #Train and print intermediate losses to console
            if ((i)%200 == 0):
                labelled_loss, unlabelled_loss, fake_loss, train_acc, disc_loss, _ , summary = \
                    sess.run([dcm.loss_d_lbl, dcm.loss_d_unl, dcm.loss_d_fake, accuracy, dcm.loss_d, d_train_op, 
                            train_summary], feed_dict=feed_dict_d)
                gen_loss, _ = sess.run([gen.loss_g, g_train_op], feed_dict=feed_dict_g)
                print("Step %d, Discriminator Loss %.4f, Generator Loss %.4f" \
                        %((i + epoch*60000//batch_size), disc_loss, gen_loss))
                print("Discriminator Loss breakdown - Labelled %.4f, Fake %.4f, Unlabelled %.4f" \
                    %( labelled_loss, fake_loss, unlabelled_loss))
                train_writer.add_summary(summary, (i + epoch*60000//batch_size))
            #Just Train
            else:
                _ = sess.run([d_train_op], feed_dict=feed_dict_d)
                _ = sess.run([g_train_op], feed_dict=feed_dict_g)

        #adversarial example evaluation
        test_probs=[]
        test_accuracies=[]
        adv_probs=[]
        adv_accuracies=[]
        detect_accuracies=[]
        print("- Evaluating on Test and Adverarial (epsilon=%.4f) Data:" %FLAGS.epsilon)
        eval_n = np.random.randint(int(x_test.shape[0]/batch_size)) #EVALUATE ON A RANDOME BATCH
        for i in range(int(x_test.shape[0]/batch_size)):
            feed_dict = {noise: np.zeros([0,100]),
                        unlabelled_images: np.zeros([0,784]),
                        labelled_images: x_test[i*batch_size: (i+1)*batch_size],
                        num_unl: 0,
                        num_lbl: batch_size,
                        labels: y_test[i*batch_size: (i+1)*batch_size]}

            if(i==eval_n):
                perturbed, disc_prob, disc_acc, disc_loss, num_correct, summary= \
                        sess.run([perturbed_images, avg_prob_lbl, accuracy, dcm.loss_d_lbl, over_half, test_summary], 
                        feed_dict=feed_dict)
                test_writer.add_summary(summary, epoch*60000//batch_size)
            else:
                perturbed, disc_prob, disc_acc, num_correct, disc_loss = \
                        sess.run([perturbed_images, avg_prob_lbl, accuracy, over_half, dcm.loss_d_lbl], 
                        feed_dict=feed_dict)

            perturbed_feed = {noise: np.zeros([0,100]),
                        unlabelled_images: np.zeros([0,784]),
                        labelled_images: perturbed,
                        num_unl: 0,
                        num_lbl: batch_size,
                        labels: y_test[i*batch_size: (i+1)*batch_size]}
            if(i == eval_n):
                perturbed_prob, perturbed_acc, perturbed_loss, num_incorrect, summary= \
                        sess.run([avg_prob_lbl, accuracy, dcm.loss_d_lbl, over_half, adv_summary], 
                                feed_dict=perturbed_feed)
                adv_writer.add_summary(summary, epoch*60000//batch_size)
            else:
                perturbed_prob, perturbed_acc, perturbed_loss, num_incorrect = \
                        sess.run([avg_prob_lbl, accuracy, dcm.loss_d_lbl, over_half], feed_dict=perturbed_feed)
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