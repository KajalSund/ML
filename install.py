# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 22:32:41 2019

@author: Kajal
"""

from sklearn.tree import export_graphviz
export_graphviz(tree, out_files="tree.dot", class_names=["malignant", "benign"],
                feature_names=cancer.feature_names, impurity=False,filled=True)


import graphviz 
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)