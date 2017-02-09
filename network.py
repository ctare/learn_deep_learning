#!/usr/bin/python3
import numpy as np
from funcs import sigmoid_f
def init_network():
    network = {}
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["w1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["w2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b3"] = np.array([0.1, 0.2])
    network["w3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    return network


def forward(network, x):
    w1, w2, w3 = [network["w%d" %i] for i in range(1, 3 + 1)]
    b1, b2, b3 = [network["b%d" %i] for i in range(1, 3 + 1)]
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid_f(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid_f(a2)
    a3 = np.dot(z2, w3) + b3
    y = a3
    return y

nw = init_network()
x = np.array([1.0, 0.5])
y = forward(nw, x)
print(y)
