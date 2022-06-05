"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from .utils import entropy, utils.information_gain, utils.gini_index
import tree.utils as utils

np.random.seed(42)

class Node:
    def __init__(self, data):
        self.children = []
        self.value = data

class DecisionTree():
    def __init__(self, criterion, max_depth=5):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"utils.information_gain", "utils.gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = Node(None)
        self.ytype = None
            

    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        no_features = X.shape[1]
        if len(X.columns) == 0:
            X.columns = [f'X{i}' for i in range(no_features)]
        if not y.name:
            y.name = 'Y'

        for i in range(no_features):
            if X[X.columns[i]].dtypes == "object" or X[X.columns[i]].dtypes == "bool":
                X[X.columns[i]] = X[X.columns[i]].astype("category")

        #Discrete input Discrete output
        def desc_desc(X, y, depth):
            if depth == self.max_depth-1:
                return Node(y.value_counts().idxmax())
            if len(y.unique()) == 1:
                return Node(y.unique()[0])
            elif len(X.columns) == 0:
                return Node(y.value_counts().idxmax())
            else:
                best_attr = X.columns[np.argmax([utils.information_gain(y, X[col], self.criterion) for col in X.columns])]
                node = Node(best_attr)
                for i in X[best_attr].unique():
                    X_i = X[X[best_attr] == i]
                    y_i = y[X[best_attr] == i]
                    node.children.append(desc_desc(X_i, y_i, depth + 1))
                return node  
        
        #Discrete input Real output
        def desc_real(X, y, depth):
            if depth == self.max_depth-1:
                return Node(y.mean())
            if len(y.unique()) == 1:
                return Node(y.unique()[0])
            elif len(X.columns) == 0:
                return Node(y.mean())
            else:
                weighted_var = []
                for i in X.columns:
                    var = np.array(pd.concat([X[i], y], axis=1).groupby(i).var()[y.name])
                    weights = np.array(y.value_counts())
                    weighted_var.append(np.multiply(var, weights))
                    best_attr = X.columns[np.argmin(weighted_var)]
                self.root = Node(best_attr)
                for i in X[best_attr].unique():
                    X_i = X[X[best_attr] == i]
                    y_i = y[X[best_attr] == i]
                    self.root.children.append(desc_real(X_i, y_i, depth + 1))
                return self.root
        
        #Real input Discrete output
        def real_desc(X, y, depth):
            if depth == self.max_depth-1:
                return Node(y.value_counts().idxmax())
            if len(y.unique()) == 1:
                return Node(y.unique()[0])
            elif len(X.columns) == 0:
                return Node(y.value_counts().idxmax())
            else:
                std = pd.concat([X, y], axis=1).groupby(y.name).agg(np.std, ddof=1).sum(axis = 0, skipna = True).to_list()
                split_attr = X.columns[np.argmin(std)]
                split_val = pd.concat([X[split_attr], y], axis = 1).groupby(y.name).agg(np.std, ddof=1)[split_attr].to_list()
                split = pd.concat([X[split_attr], y], axis = 1).groupby(y.name).mean()[split_attr].iloc[np.argmin(split_val)] + pd.concat([X[split_attr], y], axis = 1).groupby(y.name).std()[split_attr].iloc[np.argmin(split_val)]
                self.root = Node([X[split_attr].name, split])
                self.root.children.append(real_desc(X[X[split_attr] <= split], y[X[split_attr] <= split], depth + 1))
                self.root.children.append(real_desc(X[X[split_attr] > split], y[X[split_attr] > split], depth + 1))
                return self.root

        #Real input Real output
        def real_real(X, y, depth):
            if depth == self.max_depth-1:
                return Node(y.mean())
            if len(y.unique()) == 1:
                return Node(y.unique()[0])
            elif len(X.columns) == 0:
                return Node(y.mean())
            else:
                #print(y)
                std = pd.concat([X, y], axis=1).groupby(by = y.name).agg(np.std, ddof=1).sum(axis = 0, skipna = True).to_list()
                split_attr = X.columns[np.argmin(std)]
                df = pd.concat([X[split_attr], y], axis=1).sort_values(by = split_attr)
                loss = []
                for i in range(len(df)):
                    C1 = df[df[split_attr] <= df[split_attr].iloc[i]][y.name].mean()
                    C2 = df[df[split_attr] > df[split_attr].iloc[i]][y.name].mean()
                    loss.append(sum([(i-C1)**2 for i in df[df[split_attr] <= df[split_attr].iloc[i]][y.name]]) + sum([(i-C2)**2 for i in df[df[split_attr] > df[split_attr].iloc[i]][y.name]]))
                split = df[split_attr].iloc[np.argmin(loss)]
                self.root = Node([X[split_attr].name, split])
                self.root.children.append(real_real(X[X[split_attr] <= split], y[X[split_attr] <= split], depth + 1))
                self.root.children.append(real_real(X[X[split_attr] > split], y[X[split_attr] > split], depth + 1))
                return self.root
        
        X_cat = True
        for i in X.columns:
            if X[i].dtypes == "category":
                X_cat = False
                break

        if y.dtypes == "object" or y.dtypes == "category":
            self.ytype = "discrete"
            #print("Discrete output")
            if not X_cat:
                self.root = desc_desc(X, y, 0)
            else:
                self.root = real_desc(X, y, 0)
        else:
            self.ytype = "real"
            if not X_cat:
                self.root = desc_real(X, y, 0)
            else:
                self.root = real_real(X, y, 0)

    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        """
        def desc(root):
            if len(root.children) == 0:
                return root
            root = root.children[X[root].unique().index(X[root].iloc[i])]
            print(type(root))
            desc(root)
            return root
        
        def real(root):
            if len(root.children) == 0:
                return root #.value[1]
            if len(root.children) == 1:
                return root.children[0]
            root = root.children[0 if X[root.value[0]].iloc[i] <= root.value[1] else 1]
            real(root)
            return root
                
        
        root = self.root
        print(self.ytype)
        X_cat = True
        for i in X.columns:
            if X[i].dtypes == "category":
                X_cat = False
                break
        y = []  
        if not X_cat: #and (X.all().dtypes == "object" or X.all().dtypes == "category"):
            for i in range(len(X)):
                y.append(desc(root).value)
        else:  
            for i in range(len(X)):
                y.append(real(root).value)
        
        return pd.Series(y)

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        print("The list at each node has first elemnt as attribute name and second element as split value, if its real values")
        def plot_tree(root, depth):
            if len(root.children) == 0 or depth == self.max_depth:
                return
            for i in range(len(root.children)):
                print(' '*depth + str(root.value) + ': ' + str(root.children[i].value))
                plot_tree(root.children[i], depth + 1)
        plot_tree(self.root, 0)
