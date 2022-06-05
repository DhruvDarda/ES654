import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from MultiClassLogesticRegression import MultiClassLogesticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold as SKF
np.random.seed(42)


X = load_iris().data
Y = load_iris().target
ones = np.ones((X.shape[0], 1))
x_p_p = np.hstack((ones, X))
onehot_encoder = OneHotEncoder(sparse=False)


skf = SKF(n_splits=5)
skf.get_n_splits(x_p_p, Y)
for train_index, test_index in skf.split(x_p_p, Y):
    X_train, X_test = x_p_p[train_index], x_p_p[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]


model = MultiClassLogesticRegression()
model.fit(X_train, Y_train)

model.loss_v_i_plot()

# predict
Y_test_pred = model.predict(X_test)
Y_train_pred = model.predict(X_train)

test_accuracy = accuracy_score(Y_test, Y_test_pred)
train_accuracy = accuracy_score(Y_train, Y_train_pred)

print(f"train accuracy on iris is {train_accuracy}")
print(f"test accuracy on iris is {test_accuracy}")

# Parameters
num_classes = 3
color_plot = "ryb"
plot_step = 0.02

iris = load_iris()

for idx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    x1 = X_train[:, pair]
    y1 = Y_train

    # Train
    clf = model.fit(x1, y1)

    # Plot the decision boundary
    plt.subplot(2, 3, idx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # Plot the training points
    for i, color in zip(range(num_classes), color_plot):
        idx = np.where(y1 == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

plt.suptitle("Decision surface of decision trees trained on pairs of features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")
plt.show()