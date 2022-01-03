import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

binary_cmap = ListedColormap(["#FF0000", "#0000FF"])

def plot_binary_data(x, y):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x[:,0], x[:,1], c=y, cmap=binary_cmap)
    plt.show()

def plot_accuracy(epochs, accuracy):
    epochs = np.arange(epochs)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(epochs, accuracy)
    ax.set_xlabel('Epoch no.')
    ax.set_ylabel('Training accuracy')
    plt.show()

def output_per_sample(y_true, y_hat, cost):
    for (y, pred, c) in zip(y_true, y_hat, cost):
        print(f"{y = }, {pred = }, cost = {c}")
