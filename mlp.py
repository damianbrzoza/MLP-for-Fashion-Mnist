import numpy as np
import tqdm


class MLP:

    nn_architecture = [
        {"input_dim": 1296, "output_dim": 100, "activation": "relu"},
        {"input_dim": 100, "output_dim": 10, "activation": "softmax"},
    ]

    def __init__(self, seed=99, verbose=False, restore=False):
        """
        Initialize MLP object

        :param seed: seed for random
        :type seed: int
        :param verbose: If want to print help message during using MLP object
        :type verbose: bool
        :param restore: Restore from checkpoint or not?
        :type restore: bool
        """
        np.random.seed(seed)
        self.early_stoping = 0
        self.restore = restore
        self.verbose = verbose
        self.params_values = {}
        self.cost_history = []
        self.accuracy_history = []
        self.curr_out = np.array(list())
        self.yhat = 0
        self.memory = {}
        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            self.params_values['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * 0.1
            self.params_values['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * 0.1

        if self.restore:
            self.load()

    @staticmethod
    def relu(z):
        """
        Function which apply relu function to current z values

        :param z: current value
        :type z: np.array
        :return: current value after activation function a_curr
        :rtype: np.array
        """
        return np.maximum(0, z)

    @staticmethod
    def relu_backward(d_a, z):
        """
        Function which apply relu function to current error value for backpropagation process

        :param d_a: current error value
        :type d_a: np.array
        :param z: current value before activation function
        :type z: np.array
        :return: error before activation function
        :rtype: np.array
        """
        d_z = np.array(d_a, copy=True)
        d_z[z <= 0] = 0
        return d_z

    @staticmethod
    def softmax(z):
        """
        Function which apply softmax function to current z values

        :param z: current value
        :type z: np.array
        :return: current value after activation function a_curr
        :rtype: np.array
        """
        e = np.exp(z).T
        out = np.zeros_like(e)
        sum_temp = e.sum(axis=1)
        for i in range(e.shape[0]):
            out[i] = e[i] / sum_temp[i]
        return out.T

    def softmax_backward(self, d_a, z):
        """
        Softmax backward propagation

        :param d_a: current error value
        :type d_a: np.array
        :param z: current value before activation function
        :type z: np.array
        :return: error before activation function
        :rtype: np.array
        """
        sig = self.softmax(z)
        return d_a * sig * (1 - sig)

    def single_layer_forward_propagation(self, a_prev, w_curr, b_curr, activation="relu"):
        """
        Function for make forward propagation through one layer in Neural Network

        :param a_prev: previous value after activation function (input)
        :type a_prev: np.array
        :param w_curr: current weights value
        :type w_curr: np.array
        :param b_curr: current bias value
        :type b_curr: np.array
        :param activation: type of activation function
        :type activation: str
        :return: a_curr and value before activation function z_curr
        :rtype: tuple
        """
        z_curr = np.dot(w_curr, a_prev) + b_curr

        if activation is "relu":
            activation_func = self.relu
        elif activation is "softmax":
            activation_func = self.softmax
        else:
            raise Exception('Non-supported activation function')

        return activation_func(z_curr), z_curr

    def full_forward_propagation(self, x):
        """
        Function for make full forward propagation through Neural Network

        :param x: training data
        :type x: np.array
        :return: None
        :rtype: None
        """
        a_curr = x

        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            a_prev = a_curr

            activ_function_curr = layer["activation"]
            w_curr = self.params_values["W" + str(layer_idx)]
            b_curr = self.params_values["b" + str(layer_idx)]
            a_curr, z_curr = self.single_layer_forward_propagation(a_prev, w_curr, b_curr, activ_function_curr)

            self.memory["A" + str(idx)] = a_prev
            self.memory["Z" + str(layer_idx)] = z_curr

        self.curr_out = a_curr

    def get_cost_value(self, targets, epsilon=1e-10) -> float:
        """
        Cross-entropy loss function

        :param targets: ground truth
        :type targets: np.array
        :param epsilon: hiperparater
        :type epsilon: float
        :return: cross_entropy value
        :rtype: float
        """
        predictions = np.clip(self.curr_out, epsilon, 1. - epsilon)
        number_of_neurons = predictions.shape[0]
        ce_loss = -np.sum(np.sum(targets * np.log(predictions + 1e-5))) / number_of_neurons
        return ce_loss

    def get_accuracy_value(self, targets: np.array) -> float:
        """
        Function to count accuracy value

        :param targets: ground truth
        :type targets: np.array
        :return: accuracy value
        :rtype: float
        """
        prediction = self.one_hot_decode(self.curr_out)
        return np.count_nonzero(prediction == targets) / targets.shape[0]

    def single_layer_backward_propagation(self, d_a_curr, w_curr, z_curr, a_prev, activation="relu"):
        """
        Function for make single layer backward propagation

        :param d_a_curr: current error value
        :type d_a_curr: np.array
        :param w_curr: current weights value
        :type w_curr: np.array
        :param z_curr: current value before activation function
        :type z_curr: np.array
        :param a_prev: previous neuron autput value
        :type a_prev: np.array
        :param activation: type of activation function
        :type activation: str
        :return: d_a_prev, d_w_curr, db_curr
        :rtype: tuple
        """
        m = a_prev.shape[1]

        if activation is "relu":
            backward_activation_func = self.relu_backward
        elif activation is "softmax":
            backward_activation_func = self.softmax_backward
        else:
            raise Exception('Non-supported activation function')

        d_z_curr = backward_activation_func(d_a_curr, z_curr)
        d_w_curr = np.dot(d_z_curr, a_prev.T) / m
        db_curr = np.sum(d_z_curr, axis=1, keepdims=True) / m
        d_a_prev = np.dot(w_curr.T, d_z_curr)

        return d_a_prev, d_w_curr, db_curr

    def full_backward_propagation(self, targets: np.array) -> dict:
        """
        Function for make full backward propagation through Neural Network

        :param targets: ground truth
        :type targets: np.array
        :return: gradients values to update weights of NN
        :rtype: dict
        """
        grads_values = {}
        targets = targets.reshape(self.curr_out.shape)

        d_a_prev = - (np.divide(targets, self.curr_out) - np.divide(1 - targets, 1 - self.curr_out))

        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]

            d_a_curr = d_a_prev

            a_prev = self.memory["A" + str(layer_idx_prev)]
            z_curr = self.memory["Z" + str(layer_idx_curr)]
            w_curr = self.params_values["W" + str(layer_idx_curr)]

            d_a_prev, d_w_curr, db_curr = self.single_layer_backward_propagation(
                d_a_curr, w_curr, z_curr, a_prev, activ_function_curr)

            grads_values["dW" + str(layer_idx_curr)] = d_w_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values

    def update(self, grads_values, learning_rate):
        """
        Update NN weights and bias

        :param grads_values: gradient value
        :type grads_values: np.array
        :param learning_rate: current learning_rate
        :type learning_rate: float
        """
        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            self.params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
            self.params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    def train(self, train_data: np.array, train_labels: np.array, epochs: int = 1000,
              learning_rate: float = 0.001) -> tuple:
        """
        Function which trains model

        :param train_data: x data for training
        :type train_data: np.array
        :param train_labels: y data for training
        :type train_labels: np.array
        :param epochs: number of epochs for training
        :type epochs: int
        :param learning_rate: hiperparameter of training
        :type learning_rate: float
        :return: (self.params_values, self.cost_history, self.accuracy_history)
        :rtype: tuple
        """
        ohe = self.one_hot_encode(train_labels)
        pbar = tqdm.tqdm(total=epochs)
        for i in range(epochs):
            if self.early_stoping > 10:
                break
            self.full_forward_propagation(train_data)
            cost = self.get_cost_value(train_labels)
            self.cost_history.append(np.mean(cost))
            accuracy = self.get_accuracy_value(train_labels)
            if self.accuracy_history:
                if accuracy > max(self.accuracy_history):
                    self.save()
                else:
                    self.early_stoping = self.early_stoping + 1
            self.accuracy_history.append(accuracy)
            grads_values = self.full_backward_propagation(ohe)
            self.update(grads_values, learning_rate)
            pbar.update(1)
        pbar.close()
        return self.params_values, self.cost_history, self.accuracy_history

    def one_hot_encode(self, labels: np.array):
        """
        Function which convert labels to vector

        :param labels: np.array of labels to convert
        :type labels: np.array
        :return: converted labels in one_hot_encode
        :rtype: np.array
        """
        num_columns = len(np.unique(labels))
        ohe = np.zeros(shape=(num_columns, labels.shape[0]))
        i = 0
        for _ in labels:
            ohe[labels[i]][i] = 1
            i = i + 1
        if self.verbose:
            print(ohe.shape)
        return ohe

    def one_hot_decode(self, labels) -> np.array:
        """
        Function which convert vector to labels

        :param labels: np.array of predicted labels to convert
        :type labels: np.array
        :return: converted labels from one_hot_encode
        :rtype: np.array
        """
        temp = np.transpose(labels)
        decode = [np.argmax(x) for x in temp]
        if self.verbose:
            print(len(decode))
        return np.array(decode)

    def test(self, x_input, ground_truth):
        """
        Function for test model

        :param x_input: test data
        :type x_input: np.array
        :param ground_truth: test labels
        :type ground_truth: np.array
        :return: test accuracy
        :rtype: float
        """

        self.full_forward_propagation(x_input)
        accuracy = self.get_accuracy_value(ground_truth)
        self.accuracy_history.append(accuracy)

        return accuracy

    def predict(self, x_input):
        """
        Funtion to predict output for given input

        :param x_input: image data
        :type x_input: np.array
        :return: predition vector
        :rtype: np.array
        """
        self.full_forward_propagation(x_input)
        y_hat_ = self.one_hot_decode(self.curr_out)
        y_hat_ = y_hat_.astype('int32')
        return y_hat_

    def save(self):
        """
        Funtion for saving model current variables to files

        """
        for key in self.params_values.keys():
            np.save(str(key), self.params_values[key])

    def load(self):
        """
        Funtion for loading model current variables to files
        
        """
        print("Loading...")
        for key in self.params_values.keys():
            self.params_values[key] = np.load(str(key) + '.npy')
