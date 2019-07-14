# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 12:56:01 2019

@author: Kajal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#give path of dataset file
dataset = pd.read_excel('blood.xlsx')
#slicing and dicing
X = dataset.iloc[2:,1].values
y = dataset.iloc[2:,-1].values
#convert X vector to matrics by reshape method
X = X.reshape(-1,1)

plt.scatter(X, y)
plt.xlabel('age')
plt.ylabel('blood pressure')
plt.title('blood pressure prediction')
plt.show()

#linear regression class
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() #create object
lin_reg.fit(X , y)

lin_reg.score(X, y)

plt.scatter(X, y)
plt.plot(X, lin_reg.predict(X),c = "r") #plot predicted values of y by lin reg
plt.xlabel('age')
plt.ylabel('blood pressure')
plt.title('blood pressure prediction')
plt.show()

lin_reg.predict([[20]])
lin_reg.predict([[26]])


#b0 and b1
lin_reg.coef_
lin_reg.intercept_

y_pred = lin_reg.predict(X)
