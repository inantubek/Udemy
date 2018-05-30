# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:10:29 2018

@author: IS96347
"""

import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

veriler = pd.read_csv("5.1.1_musteriler.csv")

x = veriler.iloc[:,3:].values

kmeans = KMeans(n_clusters=3, init="k-means++")
kmeans.fit(x)
print(kmeans.cluster_centers_)

sonuclar = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=123)
    kmeans.fit(x)
    sonuclar.append(kmeans.inertia_)
    
print(sonuclar)

plt.plot(range(1, 11), sonuclar)