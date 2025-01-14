import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Getting the MNIST dataset
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = np.reshape(train_data, (60000, 784, 1))
test_data = np.reshape(test_data, (10000, 784, 1))
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255

# Creating weights and biases for layers
w_input_to_h1 = np.random.uniform(-1, 1, (50, 784))
w_h1_to_h2 = np.random.uniform(-1, 1, (20, 50))
w_h2_to_output = np.random.uniform(-1, 1, (10, 20))
b_input_to_h1 = np.zeros((50, 1))
b_h1_to_h2 = np.zeros((20, 1))
b_h2_to_output = np.zeros((10, 1))


def ReLU(Z):
    """
    Rectified Linear Unit

    Returns maximum of z and 0
    """
    return np.maximum(Z, 0)

def softmax(Z_f):
    """
    Returns raw output values converted into probabilities
    """
    return np.exp(Z_f) / sum(np.exp(Z_f))

def one_hot(L):
    one_hot_L = np.zeros((L.size, L.max() + 1))
    one_hot_L[np.arange(L.size), L] = 1
    return one_hot_L.T

def forward_prop(w0, b0, w1, b1, w2, b2, X):
    z1 = w0.dot(X) + b0
    a1 = ReLU(z1)
    z2 = w1.dot(z1) + b1
    a2 = ReLU(z2)
    z3 = w2.dot(a2) + b2
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

