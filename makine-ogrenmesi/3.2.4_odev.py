# -*- coding: utf-8 -*-
"""
Created on Sun May 13 20:02:15 2018

@author: Ugur
"""

import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

veriler = pd.read_csv("3.2.4.1_odev_tenis.csv")

outlook = veriler.iloc[:,:1]
lb = LabelBinarizer()
outlook = lb.fit_transform(outlook)

le = LabelEncoder()

windy = veriler.iloc[:,3:4]
windy = le.fit_transform(windy)

play = veriler.iloc[:,4:]
play = le.fit_transform(play)

df_outlook = pd.DataFrame(data=outlook, columns=["overcast","rainy","sunny"])

df_temp_humi = veriler.iloc[:,1:3]

df_windy = pd.DataFrame(data=windy, columns=["windy"])

df_play = pd.DataFrame(data=play, columns=["play"])

df_veriler = pd.concat([df_windy, df_play, df_outlook, df_temp_humi], axis=1)

x_train, x_test, y_train, y_test = train_test_split(df_veriler.iloc[:,:-1], df_veriler.iloc[:,-1:], test_size=0.33, random_state=0)

lr = LinearRegression()
lr.fit(x_train, y_train)
tahmin = lr.predict(x_test)

x_l = df_veriler.iloc[:,[0,1,2,3,4,5]]
r = sm.OLS(endog=df_veriler.iloc[:,-1:], exog=x_l).fit()
print(r.summary())

x_l = df_veriler.iloc[:,[1,2,3,4,5]]
r = sm.OLS(endog=df_veriler.iloc[:,-1:], exog=x_l).fit()
print(r.summary())