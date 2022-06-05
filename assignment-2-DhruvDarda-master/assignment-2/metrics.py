import pandas as pd
import numpy as np

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    #print(y_hat.size, y.size)
    assert(y_hat.size == y.size)
    val = pd.Series(y_hat.reset_index(drop=True) == y.sort_index().reset_index(drop=True))
    return len(val[val == True])*100 / len(val)

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    y.sort_index(inplace=True)
    cls_y_hat = y_hat.iloc[y[y == cls].index]
    return len(cls_y_hat[cls_y_hat == cls]) / len(cls_y_hat)

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    y.sort_index(inplace=True)
    cls_y_hat = y_hat[y == cls]
    cls_y = y[y == cls]
    return len(cls_y_hat[cls_y_hat == cls]) / len(cls_y)

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    x = np.array(y_hat) - np.array(y)
    return np.sqrt(np.mean(x**2))
    #return np.sqrt(np.transpose(x).dot(x) / len(y))

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    x = np.array(y_hat) - np.array(y)
    return np.mean(np.abs(x))
