# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:11:55 2018

@author: Ugur
"""

import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import numpy as np

veriler = pd.read_csv("2.1_veriler.csv")
print(veriler)

ulke = veriler.iloc[:,:1].values
print(ulke)

lb = LabelBinarizer()
ulke = lb.fit_transform(ulke)
print(ulke)

cinsiyet = veriler.iloc[:,-1:].values
print(cinsiyet)

lb = LabelEncoder()
cinsiyet[:,0] = lb.fit_transform(cinsiyet[:,0])
print(cinsiyet)

sonuc1 = pd.DataFrame(data = ulke, columns = ["fr", "tr", "us"])
print(sonuc1)

sonuc2 = pd.DataFrame(data = veriler.iloc[:,1:4].values, columns = ["boy", "kilo", "yas"])
print(sonuc2)

sonuc3 = pd.DataFrame(data = cinsiyet, columns = ["cinsiyet"])
print(sonuc3)

sonuc = pd.concat([sonuc1, sonuc2], axis = 1)
print(sonuc)

x_train, x_test, y_train, y_test = train_test_split(sonuc, sonuc3, test_size = 0.33, random_state = 0)

lr = LinearRegression()
lr.fit(x_train, y_train)
tahmin = lr.predict(x_test)
print(tahmin, y_test, sep = "\n")

sonuc2 = pd.concat([sonuc, sonuc3], axis = 1)
print(sonuc2)

boy = sonuc2.iloc[:,3:4]
print(boy)

sol = sonuc2.iloc[:,:3]
print(sol)

sag = sonuc2.iloc[:,4:]
print(sag)

sonuc22 = pd.concat([sol, sag], axis = 1)
print(sonuc22)

x_train, x_test, y_train, y_test = train_test_split(sonuc22, boy, test_size = 0.33, random_state = 0)

lr = LinearRegression()
lr.fit(x_train, y_train)
tahmin = lr.predict(x_test)
print(tahmin, y_test, sep = "\n")

liste = sonuc22.iloc[:,[0,1,2,3,4,5]].astype(float)
r = sm.OLS(endog=boy, exog=liste).fit()
print(r.summary())

liste = sonuc22.iloc[:,[0,1,2,3,5]].astype(float)
r = sm.OLS(endog=boy, exog=liste).fit()
print(r.summary())




