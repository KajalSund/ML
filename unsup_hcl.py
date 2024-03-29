# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:56:21 2019

@author: Kajal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples = 300, centers = 6, cluster_std = 0.8)


plt.scatter(x[:, 0], x[:, 1])
plt.show()


import scipy.cluster.hierarchy as sch
sch.dendrogram(sch.linkage(x, method = 'ward'))

from sklearn.cluster import AgglomerativeClustering
hca = AgglomerativeClustering(n_clusters = 10)
y_pred = hca.fit_predict(x)

plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1])
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1])
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1])
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1])
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1])
plt.scatter(x[y_pred == 5, 0], x[y_pred == 5, 1])
plt.scatter(x[y_pred == 6, 0], x[y_pred == 6, 1])
plt.scatter(x[y_pred == 7, 0], x[y_pred == 7, 1])
plt.scatter(x[y_pred == 8, 0], x[y_pred == 8, 1])
plt.scatter(x[y_pred == 9, 0], x[y_pred == 9, 1])
plt.show()
