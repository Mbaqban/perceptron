import time
from dataset import *
import numpy


class Perceptron:
    def __init__(self,
                 n_inputs_n: int,
                 n_outputs_n: int,
                 sampels: list,
                 classes: list,
                 lr: float,
                 teta: float
                 ):

        # size of input layer (numbers of input nodes)
        self.n_inputs_n = n_inputs_n
        # size of output layer (numbers of output nodes)
        self.n_outputs_n = n_outputs_n
        # learning rate : smaller learning rate ->lower speed of train higher accuracy
        self.learning_rate = lr
        # teta will use in activation function
        self.teta = teta
        # samples is dataset
        self.Sampels = sampels
        # output classification
        self.Classes = classes

        # network need one weight for each input of each neuron
        # this network is fully connected network but no hidden layers so we have (n_inputs_n * n_outputs_n) weights
        self.Weights = [[0 for j in range(n_inputs_n)]
                        for i in range(n_outputs_n)]

        # and for each output neuron we have one bias
        self.Biases = [0 for i in range(n_outputs_n)]

    def sigma_function(self, inputs, weights, bias):
        """
        inputs -> inputs of chosen neuron\n
        weights -> weights of chosen neuron\n
        bias -> bias of chosen neuron\n
        sum multiplication of inputs of neuron and weight of it and return it
        """
        sum_input = bias
        for i, w in zip(inputs, weights):
            sum_input = sum_input + (i * w)

        return sum_input

        # or in advanced way but slower
        a = [i * w for i, w in zip(inputs, weights)]
        a.append(bias)
        return sum(a)

        # or with numpy
        # you should install np | pip install numpy
        return np.dot(inputs, weights) + bias

    def activation_function(self, sum_input):
        """
        sum_input -> sum_input its output of sigma_function
        return one of (1,0,-1) base on teta
        """
        if sum_input > self.teta:
            return 1
        elif sum_input < - self.teta:
            return -1
        elif - self.teta <= sum_input <= self.teta:
            return 0

        # or in advanced way
        return 1 if sum_input > self.teta else (-1 if sum_input < - self.teta else 0)

    def train(self):
        epoch = 0  # just for save epoch number

        flag = True  # network needs more train

        while flag:

            flag = False

            for sample in self.Sampels:
                # in blow for loop j is indices of weights in range 0 to len weights and t is output we expect of each output neuron
                for j, t in zip(range(self.n_outputs_n), self.Classes[sample['label']]):
                    output_of_neuron = self.activation_function(
                        self.sigma_function(
                            sample['data'], self.Weights[j], self.Biases[j])
                    )

                    # if any output neuron fire unlikely an give us wrong output we need more train
                    # this means perceptron continue training until there is no error
                    # if we choos our teta and learning rate wrong training take too much time := infinit
                    if output_of_neuron != t:
                        flag = True

                        # and the learning algorithm
                        self.Biases[j] = self.Biases[j] + \
                            (t * self.learning_rate)

                        for wi in range(self.n_inputs_n):
                            self.Weights[j][wi] = self.Weights[j][wi] + \
                                (self.learning_rate * t * sample['data'][wi])

            epoch = epoch + 1

        return {
            "epoches": epoch,
            "weights": self.Weights,
            "biases": self.Biases
        }

    def test(self):
        for i in self.Sampels:
            output_of_each_neuron = []
            for j in range(self.n_outputs_n):

                output_of_each_neuron.append(

                    self.activation_function(
                        self.sigma_function(
                            i['data'], self.Weights[j], self.Biases[j])
                    ))

            for c in self.Classes:
                if self.Classes[c] == output_of_each_neuron:
                    print(c + i['label'])
                    break
