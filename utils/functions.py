import numpy as np

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Relu(x):
    return np.maximum(0, x)

def LeakyRelu(x):
    return np.where(x > 0, x, x * 0.01)

def dSigmoid(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

def dRelu(x):
    dx = np.ones_like(x)
    dx[x <= 0] = 0
    return dx

def dLeakyRelu(x):
    return np.where(x > 0, 1, 0.01)

def binary_cross_entropy(y, y_hat):
    # works with prediction probabilities for binary classification
    m = y_hat.shape[1]
    cost = - (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T)) / m
    return np.squeeze(cost)


def binary_cross_entropy_gradient(y, y_hat):
    return - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

def binary_classification_accuracy(y, y_hat):
    predictions = np.around(y_hat).astype(int)
    return np.mean((y == predictions))
