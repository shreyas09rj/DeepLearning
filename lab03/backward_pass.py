#!/usr/bin/python
# Gradient values for each neuron in each layer

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

X = np.array([[0.3],
              [-1.2]])

Y = np.array([[1]])

W1 = np.array([
    [0.1, -1.1],
    [-0.1, 0.4],
    [0.2, 1.1]
])
W2 = np.array([
    [0.2, 0.3, 0.1],
    [-0.1, -0.2, -0.1]
])

W3 = np.array([
    [0.3, -0.3]
])

Z1 = W1 @ X
A1 = sigmoid(Z1)

Z2 = W2 @ A1
A2 = sigmoid(Z2)

Z3 = W3 @ A2
Y_hat = sigmoid(Z3)

print("Forward Pass")
print(f"A1:", A1)
print(f"A2:", A2)
print(f"Prediction:", Y_hat)

dZ3 = (Y_hat - Y) * sigmoid_derivative(Y_hat)
dZ2 = (W3.T @ dZ3) * sigmoid_derivative(A2)
dZ1 = (W2.T @ dZ2) * sigmoid_derivative(A1)
print("Backward Pass")

print("Gradient of Output Layer")
print(dZ3)
print("Gradient of Hidden Layer 2")
print(dZ2)
print("Gradient of Hidden Layer 1")
print(dZ1)
dW3 = dZ3 @ A2.T
dW2 = dZ2 @ A1.T
dW1 = dZ1 @ X.T
print("Gradient of W3")
print(dW3)
print("Gradient of W2")
print(dW2)
print("Gradient of W1")
print(dW1)







