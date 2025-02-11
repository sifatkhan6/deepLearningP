import numpy as np
import pandas as pd

# Create truth tables/dataset for logical operations
AND = pd.DataFrame({'x1': (0,0,1,1), 'x2': (0,1,0,1), 'y': (0,0,0,1)})

# Define activation function
def g(inputs, weights):
    """Simple threshold activation function"""
    return np.where(np.dot(inputs, weights) > 0, 1, 0)

# Define training function
def train(inputs, targets, weights, eta, n_iterations):
    """Train the perceptron
    Parameters:
    inputs: input data
    targets: target values
    weights: initial weights
    eta: learning rate
    n_iterations: number of training iterations
    """
    # Add bias input
    inputs = np.c_[inputs, -np.ones((len(inputs), 1))]

    for n in range(n_iterations):
        activations = g(inputs, weights)
        weights -= eta * np.dot(np.transpose(inputs), activations - targets)

    return weights

# Initialize weights and train for AND function
w = np.random.randn(3) * 1e-4  # Small random initial weights
inputs = AND[['x1','x2']]
target = AND['y']

# Train the perceptron
w = train(inputs, target, w, 0.25, 10)

# Test the perceptron
print("\nTesting AND function:")
result = g(np.c_[inputs, -np.ones((len(inputs), 1))], w)
print(result) #all done