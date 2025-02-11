import numpy as np

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input
y = np.array([[0], [1], [1], [0]])  # XOR output (target)

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

# Training function using backpropagation
def train(inputs, target, w1, b1, w2, b2, eta, n_iterations):
    for i in range(n_iterations):
        # Forward Propagation
        hidden_input = np.dot(inputs, w1) + b1
        hidden_output = sigmoid(hidden_input)
        output_input = np.dot(hidden_output, w2) + b2
        output = sigmoid(output_input)

        # Compute error (difference between predicted and actual output)
        error = target - output

        # Backpropagation
        # Output layer
        output_gradient = error * sigmoid_derivative(output)

        # Hidden layer
        hidden_error = output_gradient.dot(w2.T)
        hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)

        # Update weights and biases
        w2 += hidden_output.T.dot(output_gradient) * eta
        b2 += np.sum(output_gradient, axis=0, keepdims=True) * eta
        w1 += inputs.T.dot(hidden_gradient) * eta
        b1 += np.sum(hidden_gradient, axis=0, keepdims=True) * eta

    return w1, b1, w2, b2

# Train the network
eta = 0.1  # Learning rate
n_iterations = 10000  # Number of iterations for training
w1, b1, w2, b2 = train(X, y, w1, b1, w2, b2, eta, n_iterations)

# Test the network after training
hidden_input = np.dot(X, w1) + b1
hidden_output = sigmoid(hidden_input)
output_input = np.dot(hidden_output, w2) + b2
output = sigmoid(output_input)

# Print the results
print("Predicted XOR output:")
print(np.round(output))