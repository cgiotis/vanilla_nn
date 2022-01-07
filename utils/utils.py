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

def plot_prediction_snapshots(x, prediction_history):
    epochs = list(prediction_history.keys())
    no_snaps = len(epochs)
    rows = 2
    cols = np.ceil(no_snaps/rows)
    plt.figure(figsize=(10,7))
    for i, key in enumerate(list(prediction_history.keys())):
        y = prediction_history[key]
        ax = plt.subplot(rows, cols, i+1)
        ax.scatter(x[:,0], x[:,1], c=y, cmap=binary_cmap, label=f'epoch: {key}')
        if i < cols:
            ax.set_xticklabels([])
        if i % cols != 0:
            ax.set_yticklabels([])
    plt.show()

def plot_test_accuracy_vs_test_noise(accuracies):
    #take accuracies as a dict[noise_lvl] = [accuracy history]
    fig, ax = plt.subplots(figsize=(5,5))
    for noise_lvl, test_accuracy in accuracies.items():
        epochs = np.arange(len(test_accuracy))
        ax.plot(epochs, test_accuracy, label=f'test noise: {noise_lvl}')
    ax.legend()
    ax.set_xlabel('Epoch no.')
    ax.set_ylabel('Test set accuracy [0,1]')
    plt.show()
