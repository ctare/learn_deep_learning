#!/usr/bin/python3

from deepLearningFromScratch.dataset.mnist import *
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

import matplotlib.pyplot as plt
plt.imshow(x_train[0].reshape(28, 28))
print(t_train[0])
plt.show()
