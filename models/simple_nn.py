import numpy as np
import utils.functions as f
import utils.utils as utils

class simple_nn(object):

    def __init__(self, architecture):
        self.architecture = architecture
        self.no_layers = len(self.architecture)
        self.params = {} # trainable parameters
        self.history = []
        # np.random.seed(1)

        # initialise network parameters; sampling from normal distribution
        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            input_dim = layer["input_shape"]
            output_dim = layer["output_shape"]
            if layer_idx < self.no_layers:
                reg_factor = np.sqrt(2/input_dim)
                self.params[f"W{layer_idx}"] = np.random.randn(output_dim, input_dim) * reg_factor
                self.params[f"b{layer_idx}"] = np.random.randn(output_dim) * reg_factor
            else:
                limit = 1/np.sqrt(input_dim)
                self.params[f"W{layer_idx}"] = np.random.uniform(low=-limit, high=limit, size=(output_dim, input_dim))
                self.params[f"b{layer_idx}"] = np.random.uniform(low=-limit, high=limit, size=output_dim)

    def forward_layer(self, A_prev, W_curr, b_curr, activation="relu"):
        # forward step for some layer n
        Z_curr = (np.dot(W_curr, A_prev).T + b_curr).T

        if activation == "relu":
            activation_func = f.Relu
        elif activation == "leakyrelu":
            activation_func = f.LeakyRelu
        elif activation == "sigmoid":
            activation_func = f.Sigmoid
        else:
            raise Exception("Wrong activation function!")

        return activation_func(Z_curr), Z_curr

    def forward_propagation(self, X, log=True):
        # perform full forward propagation
        if log:
            self.output_log = {} # memory log for previous step outputs

        A_curr = X

        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            A_prev = A_curr
            activation_func = layer["activation"]
            W_curr = self.params[f"W{layer_idx}"]
            b_curr = self.params[f"b{layer_idx}"]
            A_curr, Z_curr = self.forward_layer(A_prev, W_curr, b_curr, activation_func)

            if log:
                self.output_log[f"A{idx}"] = A_prev # needed for W backpropagation
                self.output_log[f"Z{layer_idx}"] = Z_curr # needed for global backpropagation

        # the equivalent of y_hat is returned; the output of the network
        return A_curr

    def backward_layer(self, A_prev, dA_curr, W_curr, b_curr, Z_curr, activation="relu"):
        m = A_prev.shape[1]

        if activation == "relu":
            activation_delta = f.dRelu
        elif activation == "leakyrelu":
            activation_delta = f.dLeakyRelu
        elif activation == "sigmoid":
            activation_delta = f.dSigmoid
        else:
            raise Exception("Wrong activation function!")

        activation = activation_delta(Z_curr)
        # print(f"{activation = }")
        # print(f"{dA_curr = }")
        delta = dA_curr * activation_delta(Z_curr) # common dela in all partial derivatives
        # print(f"{delta = }")
        dA_prev = np.dot(W_curr.T, delta)
        dW_curr = np.dot(delta, A_prev.T) / m
        db_curr = np.sum(delta, axis=1, keepdims=False) / m

        return dA_prev, dW_curr, db_curr

    def backward_propagation(self, Y, Y_hat):
        # full backward propagation pass
        self.gradients = {} # store gradients for all parameters in given layer
        m = Y.shape[0]
        Y = Y.reshape(Y_hat.shape)

        # print(f"cost {f.binary_cross_entropy(Y, Y_hat)}")

        # cost = f.binary_cross_entropy(Y, Y_hat)
        # utils.output_per_sample(Y, Y_hat, cost)
        dA_prev = f.binary_cross_entropy_gradient(Y, Y_hat)
        # print(dA_prev)

        for layer_idx_prev, layer in reversed(list(enumerate(self.architecture))):
            layer_idx_curr = layer_idx_prev + 1
            # print(f"backward layer {layer_idx_curr}")
            activation_func = layer["activation"]
            # print(f"{activation_func = }")

            dA_curr = dA_prev
            A_prev = self.output_log[f"A{layer_idx_prev}"]
            Z_curr = self.output_log[f"Z{layer_idx_curr}"]
            W_curr = self.params[f"W{layer_idx_curr}"]
            b_curr = self.params[f"b{layer_idx_curr}"]

            dA_prev, dW_curr, db_curr = self.backward_layer(A_prev, dA_curr, W_curr, b_curr,
                                                Z_curr, activation_func)

            self.gradients[f"dW{layer_idx_curr}"] = dW_curr
            self.gradients[f"db{layer_idx_curr}"] = db_curr


    def update_params(self, learning_rate):
        # update trainable parameters
        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            w_update = learning_rate * self.gradients[f"dW{layer_idx}"]
            w_params = self.params[f"W{layer_idx}"]
            self.params[f"W{layer_idx}"] -= w_update
            b_update = learning_rate * self.gradients[f"db{layer_idx}"]
            b_params = self.params[f"b{layer_idx}"]
            self.params[f"b{layer_idx}"] -= b_update

    def train(self, X, Y, X_test, Y_test, epochs, learning_rate):
        epoch_snapshots = [0, 999, 1999, 2999, 3999, 4999]
        # epoch_snapshots = [0, 99, 199, 299, 399, 499]
        test_accuracy = []
        test_cost = []
        prediction_history = {}

        for epoch in range(epochs):
            Y_hat = self.forward_propagation(X)

            self.backward_propagation(Y, Y_hat)
            self.update_params(learning_rate)

            Y_test_hat = self.forward_propagation(X_test, log=False)
            if epoch in epoch_snapshots:
                prediction_history[epoch] = Y_test_hat

            test_accuracy.append(f.binary_classification_accuracy(Y_test, Y_test_hat))
            test_cost.append(f.binary_cross_entropy(Y_test, Y_test_hat))

            print(f"Epoch {epoch+1} of {epochs}: accuracy = {test_accuracy[-1]}")
            print(f"Epoch {epoch+1} of {epochs}: cost = {test_cost[-1]}")

        return test_accuracy, test_cost, prediction_history

    def predict(self, X):
        predictions = self.forward_propagation(X, log=False)
        predictions = np.around(predictions).astype(int)
        return predictions
