import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
import torch

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.coef_ = np.random.uniform(y.min(), y.max(), size=(X.shape[1]+1, 1))
        n = int(X.shape[0]/batch_size)
        loss_prev = np.inf
        
        for k in range(int(n_iter/n)):
            for i in range(n):
                if i<n:
                    X_batch = np.concatenate([np.ones([batch_size, 1]), X.values[i*batch_size:(i+1)*batch_size]], axis = 1)
                    y_batch = y.values[i*batch_size:(i+1)*batch_size].reshape(batch_size, 1)
                else:
                    X_batch = np.concatenate([np.ones([batch_size, 1]), X.values[i*batch_size:]], axis = 1)
                    y_batch = y.values[i*batch_size:].reshape((n-i)*batch_size, 1)
                loss = 0
                D = np.ones(self.coef_.shape, dtype=np.float64)
                for j in range(batch_size):
                    y_pred = np.dot(X_batch[j], self.coef_)
                    loss = np.sum(np.square(y_pred - y_batch))
                    if np.abs(loss_prev - loss) < 0.00001:
                        break
                    if loss_prev == np.inf:
                        loss_prev = loss
                    loss_prev = loss
                    D = (-2) * (X_batch[j].T * (y_batch[j]-y_pred)).reshape(-1, 1)
            #D = np.concatenate([np.ones([1, 1]), D], axis = 0)
            self.coef_ -= lr*D
            if lr_type == 'inverse':
                lr = lr / (i+1)


    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.coef_ = np.random.uniform(y.min(), y.max(), size=(X.shape[1]+1, 1))
        n = int(X.shape[0]/batch_size)
        loss_prev = np.inf
        
        for j in range(int(n_iter/n)):
            for i in range(n):
                if isinstance(X, pd.DataFrame):
                    if i<n:
                        X_batch = np.concatenate([np.ones([batch_size, 1]), X.values[i*batch_size:(i+1)*batch_size]], axis = 1)
                        y_batch = y.values[i*batch_size:(i+1)*batch_size].reshape(batch_size, 1)
                    else:
                        X_batch = np.concatenate([np.ones([batch_size, 1]), X.values[i*batch_size:]], axis = 1)
                        y_batch = y.values[i*batch_size:].reshape((n-i)*batch_size, 1)
                else:
                    if i<n:
                        X_batch = np.concatenate([np.ones([batch_size, 1]), X[i*batch_size:(i+1)*batch_size, :]], axis = 1)
                        y_batch = y[i*batch_size:(i+1)*batch_size].reshape(batch_size, 1)
                    else:
                        X_batch = np.concatenate([np.ones([batch_size, 1]), X[i*batch_size:]], axis = 1)
                        y_batch = y[i*batch_size:].reshape((n-i)*batch_size, 1)
                y_pred = np.dot(X_batch, self.coef_)
                loss = np.sum(np.square(y_pred - y_batch))
                if np.abs(loss_prev - loss) < 0.0001:
                    break
                if loss_prev == np.inf:
                    loss_prev = loss
                loss_prev = loss
            D = (-2) * np.dot(X_batch.T, (y_batch-y_pred))
            self.coef_ -= lr*D #np.dot(np.dot(np.linalg.inv(np.dot(X_batch.T, X_batch)), X_batch.T), y_batch-y_pred)
            if lr_type == 'inverse':
                lr = lr / (i+1)

    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.coef_ = torch.rand(X.shape[1]+1, 1, requires_grad=True, device='cpu', dtype=torch.float64)
        n = int(X.shape[0]/batch_size)

        for k in range(int(n_iter/n)):
            for i in range(n):
                if isinstance(X, pd.DataFrame):
                    if i<n:
                        X_batch = torch.tensor(np.concatenate([np.ones([batch_size, 1]), X.values[i*batch_size:(i+1)*batch_size]], axis = 1))
                        y_batch = torch.tensor(y.values[i*batch_size:(i+1)*batch_size].reshape(batch_size, 1))
                    else:
                        X_batch = torch.tensor(np.concatenate([np.ones([batch_size, 1]), X.values[i*batch_size:]], axis = 1))
                        y_batch = torch.tensor(y.values[i*batch_size:].reshape((n-i)*batch_size, 1))
                else:
                    if i<n:
                        X_batch = torch.tensor(np.concatenate([np.ones([batch_size, 1]), X[i*batch_size:(i+1)*batch_size]], axis = 1))
                        y_batch = torch.tensor(y[i*batch_size:(i+1)*batch_size])
                    else:
                        X_batch = torch.tensor(np.concatenate([np.ones([batch_size, 1]), X[i*batch_size:]], axis = 1))
                        y_batch = torch.tensor(y[i*batch_size:])
                y_pred = torch.matmul(X_batch, self.coef_)
                loss = torch.pow(y_batch-y_pred, 2)
                loss.sum().backward()
            with torch.no_grad():
                self.coef_ -= lr*self.coef_.grad
                self.coef_.grad.zero_()

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        self.coef_ = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        self.coef_ = np.concatenate([np.ones([1, 1]), self.coef_.reshape(-1, 1)], axis=0)

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        if torch.is_tensor(self.coef_):
            self.coef_ = self.coef_.detach().numpy()
        return np.dot(np.concatenate([np.ones([X.shape[0], 1]), X], axis=1), self.coef_)

    def coef_(self):
        theta = self.coef_
        return theta[1:]

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        fig = plt.figure()
        #ax = plt.axes(projection ='3d')
        y_pred = np.dot(X.T, t_0.reshape(1, -1)) + t_1.reshape(-1, 1)
        RSS = np.sum(np.square(y_pred - y))
        plt.plot(t_0, t_1, RSS, 'ro')
        plt.title('RSS = ' + str(RSS))
        return fig

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        y_pred = np.dot(X.T, t_0.reshape(1, -1)) + t_1.reshape(-1, 1)
        fig = plt.figure()
        plt.plot(X, y, 'ro')
        plt.plot(X, y_pred, 'b')
        plt.title('t_0 = ' + str(t_0) + ' t_1 = ' + str(t_1))
        return fig

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        pass
