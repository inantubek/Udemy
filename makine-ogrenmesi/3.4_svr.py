# -*- coding: utf-8 -*-
"""
Created on Sat May 19 14:56:31 2018

@author: Ugur
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

veriler = pd.read_csv("3.3.1_maaslar.csv")

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

sc = StandardScaler()
x_scale = sc.fit_transform(x)
y_scale = sc.fit_transform(y)

svr = SVR(kernel="rbf")
svr.fit(x_scale, y_scale)

plt.scatter(x_scale, y_scale, color="red")
plt.plot(x_scale, svr.predict(x_scale), color="blue")