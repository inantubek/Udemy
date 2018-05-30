# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:32:58 2018

@author: Ugur
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

veriler = pd.read_csv("3.3.1_maaslar.csv")

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

dt = DecisionTreeRegressor(random_state=0)
dt.fit(x, y)

plt.scatter(x, y, color="red")
plt.plot(x, dt.predict(x), color="blue")