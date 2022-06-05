import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

N = 3500000
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

LR1 = LinearRegression()

start1 = time.time()
LR1.fit_autograd(X, y, batch_size=20)
y_pred = LR1.predict(X)
end1 = time.time()
print('Time taken for fit_autograd: ', end1 - start1)

LR2 = LinearRegression()

start2 = time.time()
LR2.fit_normal(X, y)
y_pred = LR2.predict(X)
end2 = time.time()
print('Time taken for fit_normal: ', end2 - start2)