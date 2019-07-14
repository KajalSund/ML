# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:36:44 2019

@author: Kajal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

plt.scatter(X[y == 0, 0], X[y == 0, 1], c ="r", label = "Setosa")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c ="b", label = "Versica")
plt.scatter(X[y == 2, 0], X[y == 2, 1], c ="g", label = "Verginica")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.title('Analysis on the Iris Dataset')
plt.show()



plt.scatter(X[y == 0, 2], X[y == 0, 3], c ="r", label = "Setosa")
plt.scatter(X[y == 1, 2], X[y == 1, 3], c ="b", label = "Versica")
plt.scatter(X[y == 2, 2], X[y == 2, 3], c ="g", label = "Verginica")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.title('Analysis on the Iris Dataset')
plt.show()



plt.scatter(X[y == 0, 0], X[y == 0, 2], c ="r", label = "Setosa")
plt.scatter(X[y == 1, 0], X[y == 1, 2], c ="b", label = "Versica")
plt.scatter(X[y == 2, 0], X[y == 2, 2], c ="g", label = "Verginica")
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend()
plt.title('Analysis on the Iris Dataset')
plt.show()



plt.scatter(X[y == 0, 0], X[y == 0, 3], c ="r", label = "Setosa")
plt.scatter(X[y == 1, 0], X[y == 1, 3], c ="b", label = "Versica")
plt.scatter(X[y == 2, 0], X[y == 2, 3], c ="g", label = "Verginica")
plt.xlabel('Sepal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.title('Analysis on the Iris Dataset')
plt.show()



plt.scatter(X[y == 0, 1], X[y == 0, 2], c ="r", label = "Setosa")
plt.scatter(X[y == 1, 1], X[y == 1, 2], c ="b", label = "Versica")
plt.scatter(X[y == 2, 1], X[y == 2, 2], c ="g", label = "Verginica")
plt.xlabel('Sepal Width')
plt.ylabel('Petal Length')
plt.legend()
plt.title('Analysis on the Iris Dataset')
plt.show()



plt.scatter(X[y == 0, 1], X[y == 0, 3], c ="r", label = "Setosa")
plt.scatter(X[y == 1, 1], X[y == 1, 3], c ="b", label = "Versica")
plt.scatter(X[y == 2, 1], X[y == 2, 3], c ="g", label = "Verginica")
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.legend()
plt.title('Analysis on the Iris Dataset')
plt.show()

