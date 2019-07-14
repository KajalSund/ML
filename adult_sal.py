# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:39:38 2019

@author: Kajal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('adult.csv', names = ['age', 'workplace', 'fnlwgt', 'education', 'education-num'
                                            ,'martial-status', 'occupation', 'relationship', 'race'
                                            , 'gender', 'Capital-gain', 'Capital-loss', 'hours-per-week'
                                            , 'native-country', 'salary'], na_values = ' ?')

#Slicing and Dicing
X = dataset.iloc[:, 0:14].values
y = dataset.iloc[:, -1].values

#gives nan count
dataset.isnull().sum()

#to display null values
temp = pd.DataFrame(X[:, [1,6,13]])

#to display count of each category null values
temp[0].values_counts()
temp[1].values_counts()
temp[2].values_counts()

#fill null values with most frequent values
temp[0] = temp[0].fillna(' Private')
temp[1] = temp[1].fillna(' Prof-speciality')
temp[2] = temp[2].fillna(' United-States')

#changes in X
X[:, [1,6,13]] = temp

#convert string to numbers by labelencoder
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()

#fit method is used to identification of column
#tranform method is used to perform task
#fit_transform is for one go to do above two

#LabelEncoder on column 1-WorkClass

X[:, 1] = lab.fit_transform(X[:, 1])
#LabelEncoder on column 1-WorkClass

X[:, 3] = lab.fit_transform(X[:, 3])
#LabelEncoder on column 1-WorkClass

X[:, 5] = lab.fit_transform(X[:, 5])
#LabelEncoder on column 1-WorkClass

X[:, 6] = lab.fit_transform(X[:, 6])
#LabelEncoder on column 1-WorkClass

X[:, 7] = lab.fit_transform(X[:, 7])
#LabelEncoder on column 1-WorkClass

X[:, 8] = lab.fit_transform(X[:, 8])
#LabelEncoder on column 1-WorkClass

X[:, 9] = lab.fit_transform(X[:, 9])
#LabelEncoder on column 1-WorkClass

X[:, 13] = lab.fit_transform(X[:, 13])

#Encoding column 14
y = lab.fit_transform(y)

#to know about the values given in sparx matrix
lab.classes_

#make sparx matrix by OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
X = one.fit_transform(X)
#to represent X in object to array
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)






from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)







from sklearn.linear_model import LinearRegression
log_reg = LinearRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)
log_reg.score(X, y)



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X, y)
y_pred = log_reg.predict(X_test)
log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)
log_reg.score(X, y)
knn.score(X, y)




from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X, y)
y_pred = log_reg.predict(X_test)
log_reg.score(X_train, y_train)
log_reg.score(X_test, y_test)
log_reg.score(X, y)
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
 














#to find best parameters for any estimator
param_grid = {'n_neighbors' : [1, 2, 3, 4, 5, 6, 7, 8, 9]}

param_grid1 = [{'criterion' : ['gini', 'entropy']},
                {'max_depth' : [3,4,5,6,7,8,9]}]

from ssklearn.model_selection import GridSearchCV
grid = GridSearchCV(knn, param_grid)
grid.fit(X, y)



grid1 = GridSearchCV(dtf, param_grid1)
grid1.fit(X, y)

grid.best_estimator_
grid.best_index_
grid.best_params_
grid.best_score_



grid1.best_estimator_
grid1.best_index_
grid1.best_params_
grid1.best_score_



