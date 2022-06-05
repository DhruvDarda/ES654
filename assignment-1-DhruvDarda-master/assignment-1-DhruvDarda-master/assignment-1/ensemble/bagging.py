import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimations = []
        self.all_preds = pd.DataFrame()
        self.X_samples = []
        self.y_samples = []
        self.X = None

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        for i in range(self.n_estimators):
            sample = np.random.choice(len(X), len(X), replace=True)
            if isinstance(X, pd.DataFrame):
                X_sample = X.iloc[sample]
                self.X_samples.append(X_sample)
                y_sample = y.iloc[sample]
                self.y_samples.append(y_sample)
            else:
                X_sample = X[sample]
                self.X_samples.append(X_sample)
                y_sample = y[sample]
                self.y_samples.append(y_sample)
            self.estimations.append(self.base_estimator.fit(X_sample, y_sample))

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y = []
        if isinstance(X, pd.DataFrame):
            for estimator in self.estimations:
                y.append(estimator.predict(X))
            self.all_preds = pd.DataFrame(y)
            y = self.all_preds.mode(axis=0)
        else:
            y_pred = []
            for estimator in self.estimations:
                y_pred.append(pd.Series(estimator.predict(X)))
            self.all_preds = pd.DataFrame(y_pred)
            y = self.all_preds.mode(axis=0).iloc[0]
        return pd.Series(y)

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        if isinstance(self.X_samples[0], pd.DataFrame):
            fig1, ax = plt.subplots(1, self.n_estimators, figsize=(27, 9))
            for i in range(self.n_estimators):
                ax[i].scatter(x = self.X_samples[i].iloc[:, 0], y = self.X_samples[i].iloc[:, 1], c=self.y_samples[i], cmap='rainbow')
                ax[i].set_title(f"Estimator {i+1}")

            fig2 = plt.figure(figsize=(12, 9))
            plt.scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], c=self.all_preds.mode(axis=0), cmap='rainbow')
            plt.title("Combined")
            fig1.tight_layout()
            fig2.tight_layout()
        else:
            fig1, ax = plt.subplots(1, self.n_estimators, figsize=(27, 9))
            for i in range(self.n_estimators):
                ax[i].scatter(x = self.X_samples[i][:, 0], y = self.X_samples[i][:, 1], c=self.y_samples[i], cmap='rainbow')
                ax[i].set_title(f"Estimator {i+1}")

            fig2 = plt.figure(figsize=(12, 9))
            plt.scatter(self.X[:, 0], self.X[:, 1], c=self.all_preds.mode(axis=0), cmap='rainbow')
            plt.title("Combined")
            fig1.tight_layout()
            fig2.tight_layout()
        return [fig1, fig2]
