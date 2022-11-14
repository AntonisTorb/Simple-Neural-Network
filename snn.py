from collections.abc import Callable

import numpy as np


class SimpleNeuralNetwork:
    '''Create a Simple Neural Network instance.'''

    def __init__(self, inputs_no: int, hidden_no: int , outputs_no: int, hidden_layers_no: int = 1, activation_function: str = "tanh") -> None:
        '''
        :param inputs_no:                 Number of input nodes.
        :param hidden_no:                 Number of hidden nodes.    
        :param outputs_no:                Number of output nodes.
        :param hidden_layers_no:          Number of hidden layers. Optional with default value 1.
        :param activation_function:       Activation function to use. Options are: "tanh", "sigmoid", "ReLU". Optional with default value "tanh".
        '''

        self.inputs_no = inputs_no
        self.hidden_no = hidden_no
        self.outputs_no = outputs_no
        self.hidden_layers_no = hidden_layers_no
        self.activation_function, self.activation_derivative = self.set_activation(activation_function)

        # Initialize weights randomly in range (-1,1).
        self.weights = []
        for layer_interface in range(self.hidden_layers_no + 1):
            if layer_interface == 0:  # Input - Hidden weights. 
                self.weights.append(np.random.rand(self.hidden_no, self.inputs_no) * 2 - 1)
            elif layer_interface == self.hidden_layers_no:  # Hidden - Output weights.
                self.weights.append(np.random.rand(self.outputs_no, self.hidden_no) * 2 - 1)
            else:  # Hidden - Hidden weights if more than 1 hidden layer.
                self.weights.append(np.random.rand(self.hidden_no, self.hidden_no) * 2 - 1)

        # Initialize biases randomly in range (-1,1).
        self.biases = []
        for layer_interface in range(self.hidden_layers_no + 1):
            if layer_interface == self.hidden_layers_no:  # Hidden - Output biases.
                self.biases.append(np.random.rand(self.outputs_no, 1) * 2 - 1)
            else:  # Input - Hidden biases and Hidden - Hidden biases if more than 1 hidden layer.
                self.biases.append(np.random.rand(self.hidden_no, 1) * 2 - 1)


    def set_activation(self, activation_function: str) -> tuple[Callable, Callable]:
        '''Sets the activation function based on user input. Internal method.'''

        if activation_function.lower() == "tanh":
            return SimpleNeuralNetwork.tanh, SimpleNeuralNetwork.dtanh
        elif activation_function.lower() == "sigmoid":
            return SimpleNeuralNetwork.sigmoid, SimpleNeuralNetwork.dsigmoid
        elif activation_function.lower() == "relu":
            return SimpleNeuralNetwork.ReLU, SimpleNeuralNetwork.dReLU
        else:
            print("Invalid activation function.")
            exit(0)


    def predict(self, input: list[float]) -> list[np.ndarray]:
        '''
        Makes a prediction with the provided input, returning a list of all results from all layers.
        Output result in last index.
        '''

        inputs = np.array(input)[:, None]  # Input list to column array.
        results = []
        for index, weight in enumerate(self.weights):
            result = np.dot(weight, inputs) + self.biases[index]
            results.append(self.activation_function(result))
            inputs = results[-1]  # Results from one layer are the inputs for the next.
        return results


    def train(self, input: list[float], target: list[float], learning_rate: float = 0.1) -> float:
        '''Trains the network with the provided data and learning rate.'''

        results = self.predict(input)
        inputs = np.array(input)[:, None]  # Input list to column array.
        results = [inputs] + results  # Include the inputs in the results list for backpropagation calculations.
        targets = np.array(target)[:, None]  # Target list to column array.
        results.reverse()  # Need to start from the final output for backpropagation.
        output_error = targets - results[0]
        for index, result in enumerate(results):
            if index >= len(results) - 1:
                break
            dB = learning_rate * output_error * self.activation_derivative(result)
            dW = np.dot(
                dB, 
                np.transpose(results[index + 1])
            )
            
            output_error = np.dot(np.transpose(self.weights[len(results) - index - 2]), output_error)  # Calculate the previous layer error.
            self.biases[len(results) - index - 2] += dB  # Index calculated based on reversed results list.
            self.weights[len(results) - index - 2] += dW
        return results[0][0][0]


    @classmethod
    def sigmoid(cls, x: float):
        return 1 / (1 + np.exp(-x))


    @classmethod
    def dsigmoid(cls, x: float):
        return SimpleNeuralNetwork.sigmoid(x) * (1 - SimpleNeuralNetwork.sigmoid(x))


    @classmethod
    def tanh(cls, x: float):
        return np.tanh(x)


    @classmethod
    def dtanh(cls, x: float):
        return 1 - (np.tanh(x) * np.tanh(x))


    @classmethod
    def ReLU(cls, x: float):
        return np.maximum(x, 0)


    @classmethod
    def dReLU(cls, x: float):
        return x > 0  