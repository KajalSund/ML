# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:26:11 2019

@author: Kajal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from   sklearn.datasets import load_boston
dataset = load_boston()

X = dataset.data
y = dataset.target

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

lin_reg.score(X,y)
lin_reg.coef_
lin_reg.intercept_

y_pred = lin_reg.predict(X)
