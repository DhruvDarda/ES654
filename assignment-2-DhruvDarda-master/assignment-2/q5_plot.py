import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
#from sklearn.linear_model import LinearRegression

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

lr = LinearRegression()

plt.figure(figsize=(15,10))
plt.plot(x,y,'o')
for i in range(1, 10):
    poly = PolynomialFeatures(degree=i)
    X = poly.transform(x.reshape(-1,1))
    lr.fit_normal(X,y)
    #lr.fit(X,y)
    y_hat = lr.predict(X)
    plt.plot(x, y_hat, label='degree = ' + str(i))
plt.legend()
plt.title('Polynomial Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

coef = []
for i in range(1, 5):
    poly = PolynomialFeatures(degree=i)
    X = poly.transform(x.reshape(-1,1))
    lr.fit_vectorised(X,y, batch_size=5)
    coef.append(lr.coef_.max())
plt.plot(range(1,5), coef) 
plt.title('Max Coefficient vs Degree') 
plt.xlabel('Degree')
plt.ylabel('Max Coefficient')  
plt.show()