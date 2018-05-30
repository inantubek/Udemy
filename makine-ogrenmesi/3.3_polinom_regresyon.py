# -*- coding: utf-8 -*-
"""
Created on Sat May 19 13:36:24 2018

@author: Ugur
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

veriler = pd.read_csv("3.3.1_maaslar.csv")

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

lr = LinearRegression()
lr.fit(x, y)

plt.scatter(x, y, color="red")
plt.plot(x, lr.predict(x), color="blue")
plt.show()

pf = PolynomialFeatures(degree=2)
x_poly = pf.fit_transform(x)
lr1 = LinearRegression()
lr1.fit(x_poly, y)

plt.scatter(x, y, color="red")
plt.plot(x, lr1.predict(x_poly), color="blue")
plt.show()

pf = PolynomialFeatures(degree=4)
x_poly = pf.fit_transform(x)
lr1 = LinearRegression()
lr1.fit(x_poly, y)

plt.scatter(x, y, color="red")
plt.plot(x, lr1.predict(x_poly), color="blue")
plt.show()