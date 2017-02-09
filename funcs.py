#!/usr/bin/python3
import numpy as np

def step_f(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid_f(x):
    return 1 / (1 + np.exp(-x))


def relu_f(x):
    return np.maximum(0, x)


def b_softmax_f(a):
    exp_a = np.exp(a)
    y = exp_a / np.sum(exp_a)
    return y


def softmax_f(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    y = exp_a / np.sum(exp_a)
    # y < 1
    # sum(y) == 1
    # -> y = 確率と言える

    # yの大小関係が変わらない
    # 推論フェーズ　大きいものを結果とするので、省略
    # 学習フェーズ
    return y


import matplotlib.pylab as plt
if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid_f(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

