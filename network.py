import numpy as np
import models.simple_nn as nn
from utils import datasets
from utils import utils

EPOCHS = 5000
LEARNING_RATE = 0.01
np.random.seed(10)

#good large nn seeds: 10 (5k epochs)

#good small nn seeds: 5, (5k epochs)

hidden_activation = "leakyrelu"
output_activation = "sigmoid"
architecture = [
    {"input_shape": 2, "output_shape": 4, "activation": hidden_activation},
    {"input_shape": 4, "output_shape": 6, "activation": hidden_activation},
    {"input_shape": 6, "output_shape": 6, "activation": hidden_activation},
    {"input_shape": 6, "output_shape": 4, "activation": hidden_activation},
    {"input_shape": 4, "output_shape": 1, "activation": output_activation}
]

# architecture = [
#     {"input_shape": 2, "output_shape": 5, "activation": hidden_activation},
#     {"input_shape": 5, "output_shape": 5, "activation": hidden_activation},
#     {"input_shape": 5, "output_shape": 1, "activation": output_activation}
# ]

net = nn.simple_nn(architecture)

x, y = datasets.generate_circle_data(10000, noise=0.05, random_state=99)

noise_lvls = [0.05, 0.1, 0.15, 0.2]
test_accuracies = {}
for noise in noise_lvls:
    x_test, y_test = datasets.generate_circle_data(500, noise=noise, random_state=11)
    test_accuracy, test_cost, prediction_history = net.train(x.T, y, x_test.T, y_test, EPOCHS, LEARNING_RATE)
    test_accuracies[noise] = test_accuracy
    utils.plot_prediction_snapshots(x_test, prediction_history)
utils.plot_test_accuracy_vs_test_noise(test_accuracies)
