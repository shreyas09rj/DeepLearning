#!/usr/bin/python
# Forward pass

# import numpy as np
#
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
#
# X = np.array([[0.3],
#               [-1.2]])
#
# W1 = np.array([
#     [0.1,-1.1],
#     [-0.1,0.4],
#     [0.2,1.1]
# ])
#
# W2 = ([
#     [0.2,0.3,0.1],
#     [-0.1,-0.2,-0.1]
#     ])
# W3 = np.array([
#     [0.3,-0.3]
# ])
#
# Z1 = W1 @ X
# A1 = sigmoid(Z1)
#
# Z2 = W2 @ A1
# A2 = sigmoid(Z2)
#
# Z3 = W3 @ A2
# Y_hat = sigmoid(Z3)
#
# print("A1 ", A1)
# print("A2 ", A2)
# print("Final Output ", Y_hat)



import numpy as np

np.random.seed(42)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

input_neurons = int(input("Enter number of input neurons: "))
hidden_layers = int(input("Enter number of hidden layers: "))

layers = [input_neurons]

for i in range(hidden_layers):
    neurons = int(input(f"Enter number of neurons in Hidden Layer {i+1}: "))
    layers.append(neurons)

output_neurons = int(input("Enter number of output neurons: "))
layers.append(output_neurons)

print("\nNetwork Architecture:")
print(layers)

# Random input vector
x = np.random.randn(input_neurons)

print("\nInput Vector:")
print(x)

# Create weight matrices
weights = []

for i in range(len(layers) - 1):
    W = np.random.randn(layers[i + 1], layers[i])
    weights.append(W)

activation = x

print("\n========== Forward Pass ==========")

for i in range(len(weights)):
    z = np.dot(weights[i], activation)
    activation = sigmoid(z)
    if i == len(weights) - 1:
        print("Output Layer")
    else:
        print(f"Hidden Layer {i+1}")

    print("Weights:")
    print(weights[i])

    print("z =")
    print(z)

    print("Activation (Sigmoid) =")
    print(activation)

print("\nFinal Prediction (ŷ):")
print(activation)