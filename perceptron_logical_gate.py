#!/usr/bin/python3

import numpy as np

def pyAND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 1.0
    return w1 * x1 + w2 * x2 >= theta


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    return np.sum(x * w) + b > 0


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    return np.sum(x * w) + b > 0


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    return np.sum(x * w) + b > 0


def XOR(x1, x2):
    return AND(OR(x1, x2), NAND(x1, x2))

print(XOR(0, 0))
