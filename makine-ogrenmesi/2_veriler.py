# -*- coding: utf-8 -*-
"""
Created on Sat May 12 20:03:11 2018

@author: Ugur
"""

import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split

veriler = pd.read_csv("2.3.1_eksikveriler.csv")
print(veriler)

numerik_veri = veriler.iloc[:,1:4].values
print(numerik_veri)

# eksik verileri işleme
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
numerik_veri = imputer.fit_transform(numerik_veri)
print(numerik_veri)

# kategorik -> numerik
ulke = veriler.iloc[:,0:1].values
print(ulke)

le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

ohe = OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

# numpy to dataframe
sonuc = pd.DataFrame(data = ulke, columns = ["fr", "tr", "us"])
print(sonuc)

sonuc2 = pd.DataFrame(data = numerik_veri, columns = ["boy", "kilo", "yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet, columns=['cinsiyet'])
print(sonuc3)

# dataframe birleştirme
s = pd.concat([sonuc, sonuc2], axis=1)
print(s)

s2 = pd.concat([s, sonuc3], axis=1)
print(s2)

# verilerin eğitim ve test için bölünmesi
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size = 0.33, random_state = 0)

# verilerin ölçeklenmesi
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_text = sc.fit_transform(x_test)