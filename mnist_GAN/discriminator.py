import tensorflow as tf
import nn_helper

MNIST_DIMENSION = 28**2
class MNIST_discriminator():
    def __init__(self, fake_images, unlabelled_images, labelled_images, num_unl, num_lbl):
        """Discriminator network with 5 hidden layers and gaussian noise
        Args:
            fake_images: generated images from the output of generator
            unlabelled_images: tf placeholder for unlabelled images
            labelled_images: tf placeholder for labelled images
            num_unl: number of generated and unlabelled data
            num_lbl: number of labelled data
        Object Fields:
            self.d5: output from the last hidden layer
            self.d_output: output logit from the discriminator
            self.logits_unl: part of d_output associated with unlabelled data
            self.logits_fake: part of d_output associated with generated data
            self.logits_lbl: part of d_output associated with labelled data
        """
        with tf.variable_scope("discriminator"):
            with tf.variable_scope("hidden1") as scope:
                d1 = nn_helper.fully_connected(
                        nn_helper.gaussian_noise(tf.concat([unlabelled_images, fake_images, labelled_images],0), sigma=0.3),
                        MNIST_DIMENSION, 1000, train_scale=False, scope=scope)
            with tf.variable_scope("hidden2") as scope:
                d2 = nn_helper.fully_connected(nn_helper.gaussian_noise(d1, sigma=0.5), 1000, 500, train_scale=False, scope=scope)
            with tf.variable_scope("hidden3") as scope:
                d3 = nn_helper.fully_connected(nn_helper.gaussian_noise(d2, sigma=0.5), 500, 500, train_scale=False, scope=scope)
            with tf.variable_scope("hidden4") as scope:
                d4 = nn_helper.fully_connected(nn_helper.gaussian_noise(d3, sigma=0.5), 500, 250, train_scale=False, scope=scope)
            with tf.variable_scope("hidden5") as scope:
                self.d5 = nn_helper.fully_connected(nn_helper.gaussian_noise(d4, sigma=0.5), 250, 250, train_scale=False, scope=scope)
            with tf.variable_scope("output") as scope:
                # We only need 10 since we're fixing the logits of the "fake" category to be 0
                self.d_output = nn_helper.fully_connected(nn_helper.gaussian_noise(self.d5, sigma=0.5), 250, 10, None, scope=scope)
                self.logits_unl, self.logits_fake, self.logits_lbl = tf.split(self.d_output, [num_unl, num_unl, num_lbl], 0)

    def calculate_loss(self, num_lbl, labels):
        """Calculate the loss for discriminator, including loss from fake, unlabeled and labeled data
        Args:
            logits_unl: unlabelled logits from discriminator logit output
            logits_fake: fake logits from discriminator logit output
            logits_lbl: labelled logits from discriminator logit output
            num_lbl: number of labeled data
            labels: the labels for labelled data
        """
        #TODO: Historical Averaging
        self.loss_d_unl = 0.5 * tf.reduce_mean(- tf.log(tf.reduce_sum(tf.exp(self.logits_unl), axis=1) + 1e-8) \
                    + tf.log(tf.reduce_sum(tf.exp(self.logits_unl),axis=1)+1))
        self.loss_d_fake = 0.5 * tf.reduce_mean(tf.log(tf.reduce_sum(tf.exp(self.logits_fake),axis=1)+1))
        self.loss_d_lbl = tf.cond(tf.greater(num_lbl, 0), 
                            lambda: tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_lbl, 
                                                                        labels=tf.one_hot(labels,10))),
                            lambda: tf.constant(0.))
        self.loss_d = self.loss_d_unl + self.loss_d_fake + self.loss_d_lbl

    def train(self, loss_d, max_gradient_norm, learning_rate):
        """Build train op for discriminator
        Args:
            loss_d: loss function for discriminator from calculate_loss
            max_gradient_norm: max clip value for gradient, part of gradient clipping preventing vanishing/exploding gradient
            learning_rate: learning rate for discriminator
        Returns:
            train op for discriminator
        """
        d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
        
        gvs = d_optimizer.compute_gradients(loss_d, 
                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="discriminator"))
        clipped_gradients=[(tf.clip_by_norm(grad, max_gradient_norm), var) for grad, var in gvs]
        d_train_op = d_optimizer.apply_gradients(clipped_gradients)
        return d_train_op
