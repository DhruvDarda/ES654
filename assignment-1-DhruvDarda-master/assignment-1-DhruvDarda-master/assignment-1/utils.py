import numpy as np
def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    counts = Y.value_counts().to_list()
    entropy_v = 0.0
    for count in counts:
        if count != 0:
            entropy_v -= count / len(Y) * np.log2(count / len(Y))
    return entropy_v

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    counts = Y.value_counts().to_list()
    gini = 1.0
    for count in counts:
        if count != 0:
            gini -= (count / len(Y)) ** 2
    return gini

def information_gain(Y, attr, criterion):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    if criterion == 'information_gain':
        total_entropy = entropy(Y)
        for i in attr.unique():
            Y_i = Y[attr == i]
            total_entropy -= len(Y_i) / len(Y) * entropy(Y_i)
        return total_entropy
    else:
        gini = 1
        for i in attr.unique():
            Y_i = Y[attr == i]
            gini -= len(Y_i) / len(Y) * gini_index(Y_i)
        return gini
