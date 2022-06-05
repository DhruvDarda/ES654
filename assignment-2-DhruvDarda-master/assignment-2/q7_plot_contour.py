import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import torch
torch.manual_seed(10)

x = np.array([i*np.pi/180 for i in range(60,300,4)]).reshape(-1,1)
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x**2 + 7 + np.random.normal(0,3,len(x)).reshape(-1,1)

'''
lr = LinearRegression()
lr.fit_normal(x.reshape(1,-1),y)
y_hat = lr.predict(x.reshape(-1,1))
plt.plot(x, y, 'ro')
plt.plot(x, y_hat, 'b')

plt.show()
'''

lr = 0.01

coef = np.random.uniform(0, 5, size=[x.shape[1]+1,1])
loss_prev = np.inf

x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
for i in range(10):
  plt.figure(figsize=(10,10))
  plt.plot(x[:, 1:], y, 'ro')
  y_pred = np.dot(x,coef)
  D = (2/x.shape[0]) * np.sum(np.dot(x.T, y_pred - y))
  loss = np.sum(np.square(y_pred - y))
  if np.abs(loss_prev - loss) < 0.0001:
    print(i, 'Converged')
    break
  if loss_prev == np.inf:
    loss_prev = loss
  coef = coef - lr * D
  loss_prev = loss
  plt.plot(x[:, 1:], y_pred, label = 'Loss: ' + str(np.sqrt(loss)))
  plt.legend()
  plt.title('Iteration: ' + str(i))
  plt.savefig('lr_'+str(i)+'.png')