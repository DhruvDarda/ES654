from .base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=10):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.no_trees = 10
        self.estimator = DecisionTreeClassifier(criterion='gini', max_depth=10) #DecisionTree(criterion=criterion, max_depth=max_depth)
        self.estimators = []
        self.all_preds = pd.DataFrame()
        self.X_samples_cols = []
        self.X_samples = []
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X_samples_cols = [np.random.permutation(np.array(range(1, X.shape[1])))[:i] for i in np.random.choice(np.array(range(1, X.shape[1])), size = self.no_trees, replace = True)]
        for i in self.X_samples_cols:
            if isinstance(X, pd.DataFrame):
                X_i = X.iloc[:,i]
                #print(X_i.shape)
            else:
                X_i = X[:,i]
                #print(X_i.shape)
            self.X_samples.append(X_i)
            self.estimators.append(self.estimator.fit(X_i, y))
        self.y = y        

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y = []
        X = X.reset_index(drop=True)
        for i in range(self.n_estimators):
            y.append(self.estimators[i].predict(X[self.X_samples_cols[i]]))
        self.all_preds = pd.DataFrame(y)
        y = self.all_preds.mode(axis=0).iloc[0]
        self.X = X
        return pd.Series(y)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        for i in range(self.n_estimators):
            self.estimators[i].plot()
        fig1, ax = plt.subplots(1, self.n_estimators, figsize=(27, 9))
        for i in range(self.n_estimators):
            ax[i].scatter(x = self.X_samples[i].iloc[:, 0], y = self.X_samples[i].iloc[:, 1], c=self.y, cmap='rainbow')
            ax[i].set_title(f"Estimator {i+1}")

        fig2 = plt.figure(figsize=(12, 9))
        plt.scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], c=self.all_preds.mode(axis=0), cmap='rainbow')
        plt.title("Combined")
        fig1.tight_layout()
        fig2.tight_layout()



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=10):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.no_trees = 10
        self.estimator = DecisionTree(criterion=criterion, max_depth=max_depth)
        self.estimators = []
        self.all_preds = pd.DataFrame()
        self.X_samples = []
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X_samples_cols = [np.random.permutation(np.array(range(1, X.shape[1])))[:i] for i in np.random.choice(np.array(range(1, X.shape[1])), size = self.no_trees, replace = True)]
        for i in self.X_samples_cols:
            if isinstance(X, pd.DataFrame):
                X = X.reset_index(drop=True)
                X_i = X.iloc[:,i]
                #print(X_i)
            else:
                X_i = X[:,i]
                #print(X_i)
            self.X_samples.append(X_i)
            self.estimator.fit(X_i, y)
            self.estimators.append(self.estimator)
        self.y = y  

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y = []
        X = X.reset_index(drop=True)
        for i in range(self.n_estimators):
            #print(X[self.X_samples_cols[i]])
            y.append(self.estimators[i].predict(X[self.X_samples_cols[i]]))
        self.all_preds = pd.DataFrame(y)
        y = self.all_preds.mean(axis=0).iloc[0]
        self.X = X
        return pd.Series(y)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        pass
