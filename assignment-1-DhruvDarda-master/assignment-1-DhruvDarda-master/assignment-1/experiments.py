
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and M for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)

# Test case 1
# Real Input and Real Output
N = 30
M = 5
X = pd.DataFrame(np.random.randn(N, M))
y = pd.Series(np.random.randn(N))

for i in range(M):
    plt.scatter(X.iloc[:, i], y)
#print(X.head())

for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print('Criteria :', criteria)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))

# Test case 2
# Real Input and Discrete Output

N = 30
M = 5
X = pd.DataFrame(np.random.randn(N, M))
y = pd.Series(np.random.randint(M, size = N), dtype="category")
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y, s=50, cmap='autumn')
plt.show()
for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('class :' + str(cls))
        print(' Mrecision: ', precision(y_hat, y, cls))
        print(' Recall: ', recall(y_hat, y, cls))

# Test case 3
# Discrete Input and Discrete Output

N = 30
M = 5
X, y = make_classification(n_samples = N, n_features = M, n_informative = M, n_redundant = 0, n_classes = M, n_clusters_per_class = 1, random_state = 42, class_sep=0.5)
X = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))), columns=['x1', 'x2'])
y = pd.Series(y)

for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('class :' + str(cls))
        print(' Mrecision: ', precision(y_hat, y, cls))
        print(' Recall: ', recall(y_hat, y, cls))

# Test case 4
# Discrete Input and Real Output

N = 30
M = 5
X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print('Criteria :', criteria)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))

