from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

plt.figure(figsize=(15,10))
for i in range(1, 10, 2):
  poly = PolynomialFeatures(degree=i)
  coef = []
  size = []
  for k in range(1,5):
    x = np.array([i*np.pi/180 for i in range(60,90*k,4)])
    np.random.seed(10)  #Setting seed for reproducibility
    y = 4*x + 7 + np.random.normal(0,3,len(x))
    X = poly.transform(x.reshape(-1,1))
    size.append(X.shape[0])
    lr.fit(X,y)
    coef.append(lr.coef_.max())
  plt.plot(size, coef, label='Degree: ' + str(i))
plt.title('Max Coefficient vs Dataset sizes for different Degrees') 
plt.xlabel('Dataset size')
plt.ylabel('Max Coefficient')  
plt.legend()
plt.show()