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
