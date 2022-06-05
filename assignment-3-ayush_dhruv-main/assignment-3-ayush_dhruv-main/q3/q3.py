import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
rng = np.random.RandomState(0)
X = rng.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
X = pd.DataFrame(X,columns=["X1","X2"])
y = pd.Series(y, index=X.index)
data=pd.concat([X,y],axis=1)
# print(y.head())
# print(data.size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
train_data=pd.concat([X_train,y_train],axis=1)
test_data=pd.concat([X_test,y_test],axis=1)
# print(train_data.head())
train_data.reset_index(drop=True,inplace=True)
test_data.reset_index(drop=True,inplace=True)
train_data.rename(columns = {0:'class'}, inplace = True)
test_data.rename(columns = {0:'class'}, inplace = True)
# print(train_data.head())
# print(test_data.head())

label = 'class'
print("Summary of class variable: \n", train_data[label].describe())
save_path = 'agModels-predictClass'  # specifies folder to store trained models
predictor = TabularPredictor(label=label, path=save_path).fit(train_data)


y_test = test_data[label]  # values to predict
test_data_nolab = test_data.drop(columns=[label])  # delete label column


y_pred = predictor.predict(test_data_nolab)
print("Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

print(predictor.leaderboard(test_data, silent=True)) 
