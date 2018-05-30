# -*- coding: utf-8 -*-
"""
Created on Sun May 27 13:25:55 2018

@author: Ugur
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv("2.1_veriler.csv")

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)