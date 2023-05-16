"""
Implementation of a basic forward pass with one hidden layer.
"""

import numpy as np
from sklearn.datasets import load_digits

# Load the digits dataset
digits = load_digits()

# Extract the features (input) and labels (output)
X = digits.data
y = digits.target

# Filter the data for binary classification of '5' vs '7'
X_binary = X[(y == 5) | (y == 7)]
y_binary = y[(y == 5) | (y == 7)]

# One-hot encode the binary labels
y_binary_onehot = np.zeros((len(y_binary), 2))
y_binary_onehot[np.arange(len(y_binary)), np.array(y_binary == 5).astype(int)] = 1


# Normalize the input data
X_normalized = X_binary / 16.0

# Set random seed for reproducibility
np.random.seed(42)

# Define hyperparameters
input_dim = X_normalized.shape[1]
hidden_dim = 20
output_dim = 2

# Initialize weights randomly
W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros(output_dim)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return a1, a2


_, predictions = forward(X_normalized, W1, b1, W2, b2)
