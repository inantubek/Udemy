# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:24:13 2018

@author: Ugur
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

veriler = pd.read_csv("3.7.2_maaslar_yeni.csv")

print(veriler.corr())

x = veriler.iloc[:,2:5].values
x1 = veriler.iloc[:,2:3].values
y = veriler.iloc[:,5:].values

lr = LinearRegression()

lr.fit(x, y)
print(sm.OLS(lr.predict(x), x).fit().summary())
print(r2_score(y, lr.predict(x)))

lr.fit(x1, y)
print(sm.OLS(lr.predict(x1), x1).fit().summary())
print(r2_score(y, lr.predict(x1)))


pr = PolynomialFeatures(degree=4)
x_poly = pr.fit_transform(x)
x1_poly = pr.fit_transform(x1)

lr.fit(x_poly, y)
print(sm.OLS(lr.predict(x_poly), x).fit().summary())
print(r2_score(y, lr.predict(x_poly)))

lr.fit(x1_poly, y)
print(sm.OLS(lr.predict(x1_poly), x1).fit().summary())
print(r2_score(y, lr.predict(x1_poly)))


sc = StandardScaler()
x_scale = sc.fit_transform(x)
x1_scale = sc.fit_transform(x1)
y_scale = sc.fit_transform(y)

svr = SVR(kernel="rbf")

svr.fit(x_scale, y_scale)
print(sm.OLS(svr.predict(x_scale), x_scale).fit().summary())
print(r2_score(y, svr.predict(x_scale)))

svr.fit(x1_scale, y_scale)
print(sm.OLS(svr.predict(x1_scale), x1_scale).fit().summary())
print(r2_score(y, svr.predict(x1_scale)))


dt = DecisionTreeRegressor(random_state=0)

dt.fit(x, y)
print(sm.OLS(dt.predict(x), x).fit().summary())
print(r2_score(y, dt.predict(x)))

dt.fit(x1, y)
print(sm.OLS(dt.predict(x1), x1).fit().summary())
print(r2_score(y, dt.predict(x1)))


rf = RandomForestRegressor(n_estimators=10, random_state=0)

rf.fit(x, y)
print(sm.OLS(rf.predict(x), x).fit().summary())
print(r2_score(y, rf.predict(x)))

rf.fit(x1, y)
print(sm.OLS(rf.predict(x1), x1).fit().summary())
print(r2_score(y, rf.predict(x1)))