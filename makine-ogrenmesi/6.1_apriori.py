# -*- coding: utf-8 -*-
"""
Created on Tue May 29 22:38:30 2018

@author: Ugur
"""

import pandas as pd
from apyori import apriori

veriler = pd.read_csv("6.1.1_sepet.csv", header=None)

t = list(list(str(veriler.values[i, j]) for j in range(0, 20)) for i in range(0, 7501))

kurallar = apriori(t, min_support=0.01,  min_confidence=0.2, min_lift=3, min_length=2)
print(list(kurallar))
