# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:26:21 2019

@author: Kajal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('adult.csv', names = ['age', 'workplace', 'fnlwgt', 'education', 'education-num'
                                            ,'martial-status', 'occupation', 'relationship', 'race'
                                            , 'gender', 'Capital-gain', 'Capital-loss', 'hours-per-week'
                                            , 'native-country', 'salary'], na_values = ' ?')


X = dataset.iloc[: ,0:14 ]
y = dataset.iloc[: , -1]

from sklearn.linear_model import LinearRegression
log_reg = LinearRegression()
log_reg.fit(X, y)
log_reg.score(X, y)

from sklearn.neighbors import KNeighbourClassifier
knn = KNeighbourClassifier()
knn.fit(X, y)
knn.score(X, y)


from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X, y)
n_b.score(X, y)


from sklearn.ensemble import VotingClassifier
vot = VotingClassifier([('LR', log_reg),
                        ('KNN', knn),
                        ('NB', n_b)])
vot.fit(X, y)
vot.score(X, y)



from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier()
bag.fit(X, y)
bag.score(X, y)



from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
rf.score(X, y)
 
