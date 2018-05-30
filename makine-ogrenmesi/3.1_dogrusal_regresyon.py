# -*- coding: utf-8 -*-
"""
Created on Sun May 13 16:22:23 2018

@author: Ugur
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

veriler = pd.read_csv("3.1.1_satislar.csv")
print(veriler)

aylar = veriler[["Aylar"]]
print(aylar)

satislar = veriler[["Satislar"]]
print(satislar)

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size = 0.33, random_state = 0)

sd = StandardScaler()
X_train = sd.fit_transform(x_train)
X_test = sd.fit_transform(x_test)
Y_train = sd.fit_transform(y_train)
Y_test = sd.fit_transform(y_test)

lr = LinearRegression();
lr.fit(x_train, y_train)
tahmin = lr.predict(x_test)

plt.plot(x_train.sort_index(), y_train.sort_index())
plt.plot(x_test, tahmin)