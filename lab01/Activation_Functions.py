#!/usr/bin/python
#From Scratch
# Use numpy and matplotlib

import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

plt.subplot(2,2,1)
plt.plot(z, sigmoid(z), label="Sigmoid", linewidth=3)
plt.plot(z, sigmoid_derivative(z), '--', label="Derivative", linewidth=3)
plt.title("Sigmoid")
plt.xlabel("z")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
# Min =0 Max = 1

def tanh(x):
    ex = np.exp(x)
    enx = np.exp(-x)
    return (ex - enx) / (ex + enx)
def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2
# tz = tanh_derivative(z)
# print(tz)
plt.subplot(2,2,2)
plt.plot(z, tanh(z), label="Tanh", linewidth=2)
plt.plot(z, tanh_derivative(z), '--', label="Derivative", linewidth=2)
plt.title("Tanh")
plt.xlabel("z")
plt.ylabel("Output")
plt.grid(True)
plt.legend()


def relu(x):
    return np.where(x > 0, x , 0)
def relu_derivative(x):
    return np.where(x > 0, 1 , 0)

plt.subplot(2,2,3)
plt.plot(z, relu(z), label="ReLU", linewidth=2)
plt.plot(z, relu_derivative(z), '--', label="Derivative", linewidth=2)
plt.title("ReLU")
plt.xlabel("z")
plt.ylabel("Output")
plt.grid(True)
plt.legend()

def leaky_relu(x,alpha=0.01):
    return np.where(x > 0, x , alpha*x)
def leaky_relu_derivative(x,alpha=0.01):
    return np.where(x > 0, 1 , alpha )

plt.subplot(2,2,4)
plt.plot(z, leaky_relu(z), label="Leaky ReLU", linewidth=2)
plt.plot(z, leaky_relu_derivative(z), '--', label="Derivative", linewidth=2)
plt.title("Leaky ReLU")
plt.xlabel("z")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.show()
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def softmax_derivative(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)




# (a) What are the minimum and maximum values for the functions ?
# Sigmoid 0 , 1
# Tanh -1, +1
# ReLU 0, infinity
# Leaky ReLU -infinity , +infinity
# Softmax 0,1

# (b) Is the output of the function zero-centred ?

# Sigmoid: Outputs only between 0 and 1, so it is not zero-centred.
# Tanh: Outputs between −1 and +1, making it zero-centred.
# ReLU: Outputs are either 0 or positive.
# Leaky ReLU: Although it allows small negative values, its outputs are not centred symmetrically around zero.
# Softmax: Outputs probabilities, which are always positive.

# (c) What happens to the gradient when the input values are too small or too big \
# Sigmoid :Gradient becomes almost 0 for very large positive or negative inputs.
# Tanh :Gradient is largest around x = 0 *For large positive or negative inputs, the gradient becomes nearly 0.
# ReLU :For positive inputs, gradient is 1.
# For negative inputs, gradient is 0.
# May cause the Dead ReLU problem.
# Leaky ReLU :Positive inputs have gradient 1.
# Negative inputs have a small constant gradient (e.g., 0.01).
# Prevents neurons from dying.
# Softmax :Gradient depends on all output probabilities.
# When one class dominates, gradients for other classes become very small.

# (d) What is the relationship between Sigmoid and Tanh?

# | Property           | Sigmoid | Tanh             |
# | ------------------ | ------- | ---------------- |
# | Output Range       | 0 to 1  | -1 to +1         |
# | Zero-centred       | No      | Yes              |
# | Gradient           | Smaller | Larger near zero |
# | Training Speed     | Slower  | Faster           |
# | Vanishing Gradient | Yes     | Yes              |

