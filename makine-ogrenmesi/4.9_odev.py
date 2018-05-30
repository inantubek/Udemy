# -*- coding: utf-8 -*-
"""
Created on Sun May 27 17:29:25 2018

@author: Ugur
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

veriler = pd.read_excel("4.9.2_iris.xls")

x = veriler.iloc[:,:4].values
y = veriler.iloc[:,4:].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)

print("LogR")
cm = confusion_matrix(y_test, y_pred)
print(cm)


knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("KNN")
cm = confusion_matrix(y_test, y_pred)
print(cm)


svc = SVC(kernel="rbf")
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print("SVC")
cm = confusion_matrix(y_test, y_pred)
print(cm)


gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("GNB")
cm = confusion_matrix(y_test, y_pred)
print(cm)


dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

print("DTC")
cm = confusion_matrix(y_test, y_pred)
print(cm)


rfc = RandomForestClassifier(n_estimators=10, criterion="entropy")
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print("RFC")
cm = confusion_matrix(y_test, y_pred)
print(cm)