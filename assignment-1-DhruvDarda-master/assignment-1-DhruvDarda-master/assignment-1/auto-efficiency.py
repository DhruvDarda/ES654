
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read real-estate data set

df = pd.read_csv('auto-mpg.csv')
convert_dict = {'mpg':float, 'cylinders': object, 'displacement':float, 'horsepower':float, 'weight':float, 'acceleration':float, 'model year':object, 'origin': object, 'car name': object}
df = df.astype(convert_dict)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y.astype("category"), s=50, cmap='autumn')
#plt.show()

tree = DecisionTree(criterion='information_gain')
tree.fit(X, y)
y_hat = tree.predict(X)
y_hat = pd.Series([y_hat[1] if y_hat[0] == '' else y_hat[0] for y_hat in y_hat])
print(y_hat)
#tree.plot()
print('Accuracy: ', accuracy(y_hat, y))
'''
for cls in y.unique():
    print('class :' + str(cls))
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))
'''