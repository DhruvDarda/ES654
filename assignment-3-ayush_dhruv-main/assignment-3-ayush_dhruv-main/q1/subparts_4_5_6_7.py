import matplotlib.pyplot as plt
import numpy as np

from logisticRegression import BinaryLogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.RandomState(0)
X = rng.randn(200, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
Y = np.multiply(Y,1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# creating model
LR = BinaryLogisticRegression(iteration_count=100,lamb=0.5)
thetas = LR.fit_autograd(X_train, y_train)
LR.plot_decision_boundary(X_train,y_train)
print(f"The learnt thetas are {thetas}")
y_hat_train = LR.predict(X_train)
LR.find_accuracy(y_train,y_hat_train,"train")
y_hat_test = LR.predict(X_test)
LR.find_accuracy(y_test,y_hat_test,"test")
LR.plot_loss()