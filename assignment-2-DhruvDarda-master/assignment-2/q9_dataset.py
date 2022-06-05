import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

X = np.random.rand(9, 1)
X = np.concatenate([X, (2*X[1]+3*X[3]).reshape(-1, 1)], axis=0)
y = np.random.rand(10, 1)

LR = LinearRegression()
LR.fit_autograd(X, y, batch_size=5)
y_hat = LR.predict(X)

print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))