"""
This script contains a minimalistic neural network implementation.
The model classifies if a given picture of a digit is either 5 or 7.
- One hot encoding of labels
- Mini-batch training of weights


Achived 99 % accuracy with just 10 epochs
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
learning_rate = 0.01
batch_size = 32
epochs = 10

# Initialize weights randomly
W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros(output_dim)


# Implement sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Implement forward pass
def forward(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return a1, a2


# Implement batch-wise training
for epoch in range(epochs):
    # Shuffle the data
    permutation = np.random.permutation(X_normalized.shape[0])
    X_shuffled = X_normalized[permutation]
    y_shuffled = y_binary_onehot[permutation]

    # Mini-batch training
    for i in range(0, X_normalized.shape[0], batch_size):
        X_batch = X_shuffled[i : i + batch_size]
        y_batch = y_shuffled[i : i + batch_size]

        # Forward pass
        a1, a2 = forward(X_batch, W1, b1, W2, b2)

        # Compute loss
        loss = -np.sum(y_batch * np.log(a2) + (1 - y_batch) * np.log(1 - a2))

        # Backward pass
        delta2 = a2 - y_batch
        delta1 = np.dot(delta2, W2.T) * a1 * (1 - a1)

        # Update weights
        dW2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0)
        dW1 = np.dot(X_batch.T, delta1)
        db1 = np.sum(delta1, axis=0)
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Print loss for monitoring
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

# Perform forward pass on the entire dataset
_, predictions = forward(X_normalized, W1, b1, W2, b2)

# Retrieving predicted label
binary_predictions = (predictions[:, 1] > 0.5).astype(int)

# CAlc aCcuracy
accuracy = np.mean(binary_predictions == y_binary_onehot[:, 1])

print(f"Accuracy: {accuracy:.4f}")
