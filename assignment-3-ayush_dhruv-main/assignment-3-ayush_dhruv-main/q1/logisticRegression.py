import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Autograd modules here
import torch


class BinaryLogisticRegression():

    def __init__(self, iteration_count,lamb=0):
        '''
        Explain arguments here
        '''
        self.iteration_count = iteration_count
        self.lamb = lamb
        self.coef_ = None  # Replace with numpy array or pandas series of coefficients learned using the fit
        self.plot_x_axis = []
        self.loss_y_axis = []

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(z))

    def fit_autograd(self, X, y):

        
        num_samples = X.shape[0]
        num_features = X.shape[1]
        X_ = np.concatenate(((np.ones(num_samples).reshape(-1, 1)), X), axis=1)
        theta = np.ones(num_features + 1)
        batch_size = num_samples
        for i in range(self.iteration_count):
            thetaa = torch.tensor(theta, requires_grad=True)
            loss = 0 # loss initiated
            for j in range(batch_size):
                z = (-1) * (torch.matmul(torch.tensor(X_[j][:].reshape(-1,1).T), thetaa)) #z = X*theta
                s = self.sigmoid(z) 
                 # DEFINITION OF CROSS ENTROPY LOSS
                loss = loss - ((y[j]) * torch.log(s) + (1-y[j]) * torch.log(1-s))
                # L2 norm addition to loss Lambda * Theta_T theta
            loss = loss + self.lamb * torch.matmul(thetaa.reshape(-1,1).T, thetaa)
            self.loss_y_axis.append(loss.item())
            self.plot_x_axis.append(i)
            loss.backward()
            gradient = thetaa.grad.numpy()
            lr = 0.005
            theta = theta - lr * gradient

        theta = pd.Series(theta)
        self.coef_ = theta
        return self.coef_


    def predict(self, X):
        num_samples = X.shape[0]
        X_ = np.concatenate(((np.ones(num_samples).reshape(-1, 1)), X), axis=1)
        theta = self.coef_.to_numpy()
        theta_t = theta.transpose()
        y = np.matmul(X_,theta_t)
        y = torch.tensor(y)
        why = self.sigmoid(y)
        pred = []
        for elem in why:
            if elem < 0.5:
                pred.append(0)
            else:
                pred.append(1)
        return pred

    def plot_decision_boundary(self, X, y_pred):
        fig = plt.figure()
        plt.scatter(X[:, 0] / max(X[:, 0]), X[:, 1] / max(X[:, 1]), c=y_pred)

        # xd = np.linspace(min(X), max(X))
        xd = np.linspace(-1, 1)
        yd = -(self.coef_[0] * xd)  # / self.weights_[1]
        plt.plot(xd, yd, 'k', lw=1, ls='--')
        plt.fill_between(xd, yd, -1, color='tab:blue', alpha=0.2)
        plt.fill_between(xd, yd, 1, color='tab:orange', alpha=0.2)
        #plt.savefig("decision_boundary.png")
        plt.show()

    def plot_loss(self):
        plt.plot(self.plot_x_axis, self.loss_y_axis)
        plt.xlabel("iteration_number")
        plt.ylabel("loss")
        plt.show()

    def find_accuracy(self, y, y_hat, set):
        assert len(y) == len(y_hat)
        no_correct = 0
        no_samples = len(y_hat)
        no_correct = (y_hat == y).sum()
        accuracy = float(no_correct) / float(no_samples) * 100
        print(f"Got {no_correct}/{no_samples} with accuracy {accuracy:.2f}% for {set} set")
