from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AdaBoostClassifier():
    def __init__(self, base_estimator = None, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        if base_estimator != None:
            self.base_estimator = DecisionTreeClassifier(max_depth=1)
        else:
            self.base_estimator = base_estimator
        self.n_estimator = n_estimators
        self.estimators_ = []
        self.classes = []
        self.all_preds = pd.DataFrame()
        self.X = None
        self.weights = None

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.weights = np.ones(len(y)) / len(y)
        for i in range(self.n_estimator):
            estimator = self.base_estimator
            estimator.fit(X, y, self.weights)
            y_hat = estimator.predict(X)
            error = np.sum(self.weights[y != y_hat])
            alpha = np.log((1 - error) / error)/2
            self.weights = self.weights * np.exp(alpha * (y != y_hat))
            self.weights += self.weights * np.exp(-alpha * (y == y_hat))
            self.weights = self.weights / np.sum(self.weights)
            self.estimators_.append((estimator, alpha))
        self.classes = y.unique()
        self.X = X

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        sign = []
        for i in range(self.n_estimator):
            estimator, alpha = self.estimators_[i]
            y_hat = estimator.predict(X)
            lst = [-1 if y_hat[i] == self.classes[0] else 1 for i in range(len(y_hat))]
            lst = np.array(lst)*alpha
            sign.append(lst)
        self.all_preds = pd.DataFrame(sign).T
        sign = self.all_preds.sum(axis=1).tolist()
        lst = [self.classes[0] if sign[i] == -1 else self.classes[1] for i in range(len(sign))]
        return pd.Series(lst, name='y')

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        fig1, ax = plt.subplots(1, self.n_estimator, figsize=(27, 9))
        for i in range(self.n_estimator):
            estimator, alpha = self.estimators_[i]
            ax[i].scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], c=self.all_preds.iloc[:, i], cmap='rainbow', s=self.weights*1000)
            ax[i].set_title(f"Estimator {i+1} with alpha = {alpha}")

        fig2 = plt.figure(figsize=(8, 7))
        plt.scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], c=self.all_preds.mode(axis=0), cmap='rainbow', s=self.weights*1000)
        plt.title("Combined")
        fig1.tight_layout()
        fig2.tight_layout()
        return [fig1, fig2]
