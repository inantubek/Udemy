# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:37:23 2018

@author: Ugur
"""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

veriler = pd.read_csv("5.1.1_musteriler.csv")

x = veriler.iloc[:,3:].values

ac = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")
y = ac.fit_predict(x)

plt.scatter(x[y==0, 0], x[y==0, 1], s=100, c="red")
plt.scatter(x[y==1, 0], x[y==1, 1], s=100, c="blue")
plt.scatter(x[y==2, 0], x[y==2, 1], s=100, c="green")
plt.scatter(x[y==3, 0], x[y==3, 1], s=100, c="yellow")
plt.show()

dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))