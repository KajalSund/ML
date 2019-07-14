# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:48:36 2019

@author: Kajal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


m = 100
X = 8 * np.random.randn(m, 1)
y = 2 * X ** 2 + X + np.random.randn(m, 1)


plt.scatter(X ,y)
#zoom the graph between limits
plt.axis([-3, 3 ,0 ,9])
plt.show()


#apply polynomial function to create 2nd order poly
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias = False)
X_poly = poly.fit_transform(X)


#apply linearregression algo to draw curve
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)


X_new = np.linspace(-3 ,3 ,100).reshape(-1, 1)
X_new_poly = poly.fit_transform(X_new)
y_new = lin_reg.predict(X_new_poly)

plt.scatter(X ,y)
plt.plot(X_new, y_new, c = "r")
plt.axis([-3, 3 ,0 ,9])
plt.show()

#b1 , b2
lin_reg.coef_

#b0
lin_reg.intercept_

