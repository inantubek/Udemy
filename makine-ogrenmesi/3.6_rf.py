# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:59:46 2018

@author: Ugur
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

veriler = pd.read_csv("3.3.1_maaslar.csv")

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(x, y)

plt.scatter(x, y, color="red")
plt.plot(x, rf.predict(x), color="blue")
plt.show()