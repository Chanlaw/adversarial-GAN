import tensorflow as tf

def fgsm_attack(labelled_images, softmax_loss_lbl, epsilon):
    """Perform fast gradient sign attack to labelled_images for discriminator
    Args:
        labelled_images: images being perturbed
        softmax_loss_lbl: softmax loss on labelled output percentage from discriminator
        epsilon: the constant used to multiply with the perturbation in fgsm
    """
    with tf.name_scope('adv'):
        pertubation = tf.sign(tf.gradients(softmax_loss_lbl, labelled_images))
        #apply pertubation to images
        perturbed_images = tf.squeeze(epsilon * pertubation) + labelled_images
    return pertubation, perturbed_images