import numpy as np
import models.simple_nn as nn
from utils import datasets
from utils import utils

EPOCHS = 10000
LEARNING_RATE = 0.01

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
#     {"input_shape": 5, "output_shape": 1, "activation": output_activation}
# ]

net = nn.simple_nn(architecture)

x, y = datasets.generate_circle_data(10000, noise=0.05, random_state=99)
# utils.plot_binary_data(x, y)
x_test, y_test = datasets.generate_circle_data(500, noise=0.15, random_state=11)

training_accuracy, training_cost, b_history = net.train(x.T, y, EPOCHS, LEARNING_RATE)
# utils.plot_accuracy(EPOCHS, b_history)
utils.plot_accuracy(EPOCHS, training_accuracy)
predictions = net.predict(x_test.T)
# print(predictions)
utils.plot_binary_data(x_test, predictions)
