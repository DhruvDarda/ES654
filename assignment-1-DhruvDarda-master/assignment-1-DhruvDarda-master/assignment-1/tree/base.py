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
    """
    Creates node for the tree
    """
    def __init__(self, data):
        self.children = []
        self.value = data

class DecisionTree():
    """Implemented Decision Tree Algorithm below"""
    def __init__(self, criterion, max_depth=10, weights = None):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"utils.information_gain", "utils.gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = Node(None)        

    def fit(self, X, y, weights = None):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.weights = weights
        no_features = X.shape[1]
        if len(X.columns) == 0:
            X.columns = [f'X{i}' for i in range(no_features)]
        if not y.name:
            y.name = 'Y'

        #print(f'X: {X.shape}')

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
                    weights = np.array(X[i].value_counts())
                    weights = weights / np.sum(weights)
                    if self.weights:
                        weights = weights * self.weights
                    weighted_var.append(np.sum(np.multiply(var, weights)))
                #print(weighted_var)
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
                if self.weights != None:
                    std = std * self.weights
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
                std = pd.concat([X, y], axis=1).groupby(by = y.name).agg(np.std, ddof=1).sum(axis = 0, skipna = True).to_list()
                if self.weights != None:
                    std = std * self.weights
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

        def mixed_desc(X, y, depth, categories, continuous):
            if depth == self.max_depth-1:
                return Node(y.value_counts().idxmax())
            if len(y.unique()) == 1:
                return Node(y.unique()[0])
            if len(X.columns) == 0:
                return Node(y.value_counts().idxmax())
            elif len(categories) == 0:
                X = X[continuous]
                std = pd.concat([X, y], axis=1).groupby(y.name).agg(np.std, ddof=1).sum(axis = 0, skipna = True).to_list()
                split_attr = X.columns[np.argmin(std)]
                split_val = pd.concat([X[split_attr], y], axis = 1).groupby(y.name).mean()[split_attr].to_list()
                split = pd.concat([X[split_attr], y], axis = 1).groupby(y.name).max()[split_attr].iloc[np.argmin(split_val)]
                #print(split)
                self.root = Node([X[split_attr].name, split])
                if len(X[X[split_attr] <= split]) > 0 and len(X[X[split_attr] > split]) > 0:
                    self.root.children.append(mixed_desc(X[X[split_attr] <= split], y[X[split_attr] <= split], depth + 1, categories, continuous))
                    self.root.children.append(mixed_desc(X[X[split_attr] > split], y[X[split_attr] > split], depth + 1, categories, continuous))
                return Node(y.value_counts().idxmax())
            else:
                best_attr = categories[np.argmax([utils.information_gain(y, X[col], self.criterion) for col in categories])]
                node = Node(['',best_attr])
                categories.remove(best_attr)
                for i in X[best_attr].unique():
                    X_i = X[X[best_attr] == i]
                    y_i = y[X[best_attr] == i]
                    node.children.append(mixed_desc(X_i, y_i, depth + 1, categories, continuous))
            return node 

        categorical = []
        continuous = []
        for i in X.columns:
            if X[i].dtypes == "category":
                categorical.append(i)
            else:
                continuous.append(i)
        
        for i in continuous:
            X[i] = (X[i]-X[i].min())/X[i].max()

        if y.dtypes == "object" or y.dtypes == "category":
            if len(continuous)==0:
                self.root = desc_desc(X, y, 0)
            elif len(categorical)==0:
                self.root = real_desc(X, y, 0)
            else:
                self.root = mixed_desc(X, y, 0, categorical, continuous)
        else:
            if len(continuous)==0:
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
        def desc(root, Xi):
            if len(root.children) == 0:
                return root
            if X[root.value].unique().to_list().index(Xi[root.value]) >= len(root.children):
                return root
            root = root.children[X[root.value].unique().to_list().index(Xi[root.value])]
            desc(root, Xi)
            return root
        
        def real(root, Xi):
            if len(root.children) == 0:
                return root #.value[1]
            if len(root.children) == 1:
                return root.children[0]
            if root.value[0] == '':
                root = root.children[X[root.value[1]].unique().to_list().index(Xi[root.value[1]])]
            else:
                #print(root.value[0])
                #print(Xi[root.value[0]])
                root = root.children[0 if Xi[root.value[0]] <= root.value[1] else 1]
            real(root, Xi)
            return root
                
        
        root = self.root
        X_cont = False
        if isinstance(X, pd.DataFrame):
            for i in X.columns:
                if X[i].dtypes != "category":
                    X_cont = True
                    break
        else:
            if X.dtypes != "category":
                X_cont = True
        y = []  
        if not X_cont: #and (X.all().dtypes == "object" or X.all().dtypes == "category"):
            for i in range(len(X)):
                y.append(desc(root, X.iloc[i]).value)
        else:  
            for i in range(len(X)):
                y.append(real(root, X.iloc[i]).value)
        
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
        print("The list at each node has first element as attribute name and second element as split value, if its real values")
        def plot_tree(root, depth):
            if len(root.children) == 0 or depth == self.max_depth:
                return
            for i in range(len(root.children)):
                if isinstance(root.value, list):
                    if isinstance(root.children[i].value, list):
                        print(" "*depth + 'root: ' + str(root.value[0]) + ':- ' 'child: ' + str(root.children[i].value[1]))
                    else:
                        print(" "*depth + 'root: ' + str(root.value[0]) + ':- ' 'child: ' + str(root.children[i].value))
                    plot_tree(root.children[i], depth + 1)
                else:
                    print(" "*depth + 'root: ' + str(root.value) + ':- ' + 'child: ' + str(root.children[i].value))
                    plot_tree(root.children[i], depth + 1)
        plot_tree(self.root, 0)
