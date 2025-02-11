import numpy as np
import pandas as pd

# Create truth tables/dataset for logical operations
XOR = pd.DataFrame({'x1': (0,0,1,1), 'x2': (0,1,0,1), 'y': (0,1,1,0)})

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Neural network architecture
input_layer = 2    # 2 input neurons
hidden_layer = 2   # 2 neurons in the hidden layer
output_layer = 1   # 1 output neuron

# Initialize weights and biases
np.random.seed(42)  # For reproducibility
w1 = np.random.randn(input_layer, hidden_layer)  # Weights for input to hidden layer
b1 = np.random.randn(1, hidden_layer)  # Bias for hidden layer
w2 = np.random.randn(hidden_layer, output_layer)  # Weights for hidden to output layer
b2 = np.random.randn(1, output_layer)  # Bias for output layer