import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
from numpy import linalg as la

np.random.seed(42)

class MultiClassLogesticRegression:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def of_grad(self, X, Y, W, nu):
        Z = -X @ W
        P = self.Softmax(Z, ax=1)
        N = X.shape[0]
        gd = (1 / N) * (X.T @ (Y - P)) + 2 * nu * W
        return gd

    def Softmax(self, x, ax=1):
        max = np.max(x, axis=ax, keepdims=True)  # returns max of each row and keeps same dims
        e_x = np.exp(x - max)  # subtracts each row with its max value
        sum = np.sum(e_x, axis=ax, keepdims=True)  # returns sum of each row and keeps same dims
        f_x = e_x / sum
        return f_x

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient_descent(self, X, Y, max_itr=1000, alpha=0.1, nu=0):
        self.Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1, 1))
        self.W = np.zeros((X.shape[1], self.Y_onehot.shape[1]))
        i = 0
        ilist = []
        losslist = []
        while i < max_itr:
            i = i + 1
            self.W -= alpha * self.of_grad(X, self.Y_onehot, self.W, nu)
            ilist.append(i)
            losslist.append(self.loss(X, self.Y_onehot, self.W, 0))
            # print(W)

            df = pd.DataFrame({'i': ilist, 'loss': losslist})

        return self.W, df

    def loss(self, X, Y, W, nu):
        Z = - X @ W
        N = X.shape[0]
        loss = 1 / N * (np.trace((X @ W) @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1)))) + (nu * la.norm(W, 2))
        return loss

    def fit(self, X, Y):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.W = np.random.randn(self.n_features)
        self.b = np.random.randn()
        self.W, self.lossplotinfo = self.gradient_descent(X, Y)

    def predict(self, A):
        Z = -A @ self.W
        P = self.Softmax(Z)
        return np.argmax(P, axis=1)

    def loss_v_i_plot(self):
        return self.lossplotinfo.plot(x='i',
                                      y='loss',
                                      xlabel='i',
                                      ylabel='loss')