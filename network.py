import numpy as np
import matplotlib.pyplot as plt
import yaml
import gzip
import tqdm
import asciichartpy
import pickle
import random

from timeit import default_timer as timer

# our special libraries for the network
import network
# TASK 1: Implement activation functions and their derivatives. There are several activation functions used in practice with different properties. Here, you implement the three most common functions (sigmoid, tanh, relu) [to be used in the feedfoward part of the neural network] and their derivatives [to be used in the backpropagation/gradient descent update part of the neural network]. 

class ActivationFunction:
    def __init__(self, name):
        self.name = name  # Name of the activation function (e.g., 'sigmoid', 'tanh', 'relu')
        self.last_output = None  # Cache for storing the last output to avoid redundant calculations.

    def function(self, z):
        # Apply the appropriate activation function based on the name
        if self.name == 'sigmoid':
            # Clip input to prevent overflow for large values
            z = np.clip(z, -500, 500)
            # Sigmoid function: 1 / (1 + e^(-z))
            self.last_output = 1.0 / (1.0 + np.exp(-z))
        elif self.name == 'tanh':
            # Hyperbolic tangent function
            self.last_output = np.tanh(z)
        elif self.name == 'relu':
            # Rectified Linear Unit (ReLU) function: max(0, z)
            self.last_output = np.maximum(0, z)
        else:
            raise ValueError(f"Unsupported activation function: {self.name}")
        return self.last_output

    def derivative(self, z):
        # Compute the derivative of the activation function
        if self.name == 'sigmoid':
            # Derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z))
            return self.last_output * (1 - self.last_output) #self.last_output holds function(z)
        elif self.name == 'tanh':
            # Derivative of tanh: 1-tanh(z)^2
            return 1.0 - np.tanh(z) ** 2
        elif self.name == 'relu':
            # Derivative of ReLU: if z>0 Derivative of ReLU=1 otherwise Derivative of ReLU=0
            return np.where(z > 0, 1.0, 0.0)
        else:
            raise ValueError(f"Unsupported activation function derivative: {self.name}")
            
# TASK 2: Implement the initialization, feedforward, backpropagation and gradient descent update steps. Instructions in code: there are 5 parts.

class Network:
    def __init__(self, sizes, activations=None):
        self.num_layers = len(sizes)  # Total number of layers in the network
        self.sizes = sizes  # Number of neurons in each layer
        # Part 1: Initialize biases and weights with Gaussian distribution. What should the dimensions of the two arrays be?
        self.biases = []
        for y in sizes[1:]:
            bias = np.random.randn(y, 1)
            self.biases.append(bias)
        self.weights = []
        for x, y in zip(sizes[:-1], sizes[1:]):
            weight = np.random.randn(y, x)  
            self.weights.append(weight)
        # Initialize activation functions for each layer
        if activations is None:
            # Default activation function is sigmoid if not specified
            activations = ['sigmoid'] * (self.num_layers - 1)
        elif len(activations) != (self.num_layers - 1):
            raise ValueError("Number of activation functions must match number of layers minus one.")
        self.activations = [ActivationFunction(act) for act in activations]

    def feedforward(self, a):
        # Part 2: Propagate the input forward through the network. Here, "a" is some input to the network (the image array in             our case) - once the weights are learned, output of the neural network for given input a, is feedforward(a). For                 layer j, implement a_(j+1) = activation_function_j(w.a_j + b) where w are the weights, b are the biases and a_j, a_(j+1)         represent the layers. 
        for b, w, activation in zip(self.biases, self.weights, self.activations):
            a = activation.function(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, initial_learning_rate, test_data=None, decay_rate=0.0):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        loss_history = []
        learning_rate = initial_learning_rate

        for j in range(epochs):
            if decay_rate > 0:
                learning_rate = initial_learning_rate * (1. / (1. + decay_rate * j))
        
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
        
            if test_data:
                evaluation_result = self.evaluate(test_data)
                accuracy = evaluation_result / n_test
                print(f"Epoch {j + 1}: {evaluation_result} / {n_test} - Accuracy: {accuracy:.2%}")
                loss_history.append(1 - accuracy)
        print("Loss Function vs. Epoch")
        print(asciichartpy.plot(loss_history, {'height': 14}))
    

    def update_mini_batch(self, mini_batch, learning_rate):
        # Update network weights and biases based on a single mini-batch
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # Gradient accumulator for biases
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # Gradient accumulator for weights
        for x, y in mini_batch:
            # Compute gradient for current sample
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Update gradient accumulators
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Part 3: Apply gradient descent to update weights and biases using the learning rate and batch size. In particular,             recall that for weight w, new weight is given by: w-(eta/batch size)*nabla_w
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # Gradient accumulator for biases
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # Gradient accumulator for weights
    
    # Forward pass
        activation = x
        activations = [x]  # List to store all activations, layer by layer
        zs = []  # List to store all z vectors, layer by layer
        for b, w, activation_func in zip(self.biases, self.weights, self.activations):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = activation_func.function(z)
            activations.append(activation)
    
    # Backward pass
        delta = self.cost_derivative(activations[-1], y) * self.activations[-1].derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

    # Here we go backwards, starting from the second to last layer to the first layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activations[-l].derivative(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        # Evaluate the network's performance on test data
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        # Count the number of correct predictions
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return  (output_activations - y)
def load_data():
    """
    Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    """
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        # In Python 3, 'encoding' must be specified when reading Python 2 pickled files
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    Return a tuple containing (training_data, validation_data, test_data).

    MNIST data is a 28x28 pixel image, with each pixel brightness value ranging from 0 to 1. 
    Reshape it into a 28x28 = 784 element row vector.
    """
    tr_d, va_d, te_d  = load_data()
    training_inputs   = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results  = [vectorized_result(y) for y in tr_d[1]]
    training_data     = list(zip(training_inputs, training_results)) # Explicitly convert to list
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data   = list(zip(validation_inputs, va_d[1]))        # Explicitly convert to list
    test_inputs       = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data         = list(zip(test_inputs, te_d[1]))              # Explicitly convert to list
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    Return a 10-dimensional unit vector with a 1.0 in the jth position and zeroes elsewhere.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
with open('network_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

training_data, validation_data, test_data = load_data_wrapper()
print("Loaded the MNIST database...")
# Initialize the Network with configurations from YAML file
net = network.Network(config['network_structure'], 
                      config['activation_functions'])
print("Thinking Machine Awakened...")


print("Training the NN...")
t0 = timer()
# use Stochastic Gradient Descent to optimize the network
net.SGD(training_data, 
        config['epochs'], 
        config['mini_batch_size'], 
        config['learning_rate'], 
        test_data=test_data)

print("Done Training and Testing the Machina.")
print("Training time = {:0.1f}".format(timer() - t0) + " seconds.")