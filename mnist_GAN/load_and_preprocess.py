# Data preprocessing for mnist_GAN.py

import numpy as np

def load_mnist_data():
    """Load mnist data into numpy arrays
    Returns:
        x_train: 60000 images from the mnist train and validation set
        y_train: 60000 labels indicating the 0-9 numerical representation of x_train
        x_test: 10000 images from the mnist test set
        y_test: 10000 labels for images in the mnist test set / x_test
    """
    data = np.load('mnist.npz')
    x_train = np.concatenate([data['x_train'], data['x_valid']], axis=0)
    y_train = np.concatenate([data['y_train'], data['y_valid']])
    assert x_train.shape[0] == 60000
    assert y_train.shape[0] == 60000
    x_test = data['x_test']
    y_test = data['y_test']
    assert x_test.shape[0] == 10000
    assert y_test.shape[0] == 10000
    return (x_train, y_train), (x_test, y_test)

def select_labelled_data(examples_per_class):
    """select a fix number of labeled data from the train set for each class
    Args:
        examples_per_class: number of examples generated for each class 0-9
    Returns:
        x_labelled: numpy array (examples_per_class * 10, 784) representing images representing 0 to 9
        y_labelled: (examples_per_class * 10, 1) representing labels for x_labelled
    """
    np.random.seed(2)
    (x_train, y_train), _ = load_mnist_data()

    # Put x_train and y_train in random order
    idx = np.random.permutation(x_train.shape[0]) 
    x_train = x_train[idx]
    y_train = y_train[idx]

    x_labelled = []
    y_labelled = []
    for i in range(10): # Iterate though label 0-9
        x_labelled.append(x_train[y_train==i][:examples_per_class])
        y_labelled.append(y_train[y_train==i][:examples_per_class])
    x_labelled = np.concatenate(x_labelled, axis=0)
    y_labelled = np.concatenate(y_labelled, axis=0)
    return x_labelled, y_labelled
