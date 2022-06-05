import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.utils import shuffle
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

np.random.seed(42)

# Read dataset

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

df = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))), columns=['x1', 'x2', 'y'])

shuffle_indices = np.arange(len(X))
np.random.shuffle(shuffle_indices)
train_index = shuffle_indices[:int(len(X) * 0.7)]

test_index = shuffle_indices[int(len(X) * 0.7):]

X_train = df[['x1','x2']].iloc[train_index]
y_train = df['y'].iloc[train_index]
X_test = df[['x1','x2']].iloc[test_index]
y_test = df['y'][test_index]

#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='autumn')
#plt.show()

max_depth = 10
tree = DecisionTree(criterion= 'information_gain', max_depth=max_depth)

tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)
print(y_hat)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y_train.unique():
    print('class :' + str(cls))
    print('Precision: ', precision(y_hat.reset_index(), y_test.reset_index(), cls))
    print('Recall: ', recall(y_hat.reset_index(), y_test.reset_index(), cls))

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
ax1.scatter(X_test.iloc[:,0], X_test.iloc[:,1], c=y_test, s=50, cmap='autumn')
ax2.scatter(X_test.iloc[:,0], X_test.iloc[:,1], c=y_hat, s=50, cmap=matplotlib.cm.get_cmap('autumn_r'))
plt.title('Decision Tree with max_depth = ' + str(max_depth))
ax1.title.set_text("Real Output")
ax2.title.set_text('Predicted Output')
plt.show()


kf = KFold(n_splits=5)
for i in range(5, 10):
    max_depth = i
    print('max_depth = ' + str(max_depth))
    Accuracy = []
    for train, test in kf.split(X):
        tree = DecisionTree(criterion= 'information_gain', max_depth=max_depth)
        X_train = df[['x1','x2']].iloc[train]
        X_test = df[['x1','x2']].iloc[test]
        y_train = df['y'].iloc[train]
        y_test = df['y'].iloc[test]
        tree.fit(X_train, y_train)
        y_hat = tree.predict(X_test)
        Accuracy.append(accuracy(y_hat, y_test))
    print('Accuracy: ', np.mean(Accuracy))