import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logisticRegression import BinaryLogisticRegression

rng = np.random.RandomState(0)
X = rng.randn(200, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

y = np.multiply(Y,1)
LR = BinaryLogisticRegression(iteration_count=100)
thetas = LR.fit_autograd(X, Y)
print(f"The learnt thetas are {thetas}")
y_hat = LR.predict(X)
print(f"The ground truth -> {y}")
print(f"The predicted values -> {y_hat}")
