# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:25:14 2019

@author: Kajal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.datasets import load_digits
dataset = load_digits()

X = dataset.data
y = dataset.target

some_digit = X[1234]
some_digit_image = some_digit.reshape(8, 8)

plt.imshow(some_digit_image)
plt.show()

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(max_depth = 17)
dtf.fit(X, y)

dtf.score(X, y)

dtf.predict(X[[23, 300, 456, 789, 1234], 0:64])



