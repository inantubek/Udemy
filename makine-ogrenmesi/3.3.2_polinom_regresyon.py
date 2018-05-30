#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('3.3.1_maaslar.csv')

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x), color = 'blue')
plt.show()

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.show()

#tahminler

print(lin_reg.predict(11))
print(lin_reg.predict(6.6))

print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))









    

