# -*- coding: utf-8 -*-
"""
Created on Wed May 30 16:14:41 2018

@author: IS96347
"""

import pandas as pd
import random
import matplotlib.pyplot as plt

veriler = pd.read_csv("7.1.1_Ads_CTR_Optimisation.csv")

N, d = veriler.shape

secilenler = []
toplam = 0

for n in range(0, N):
    ad = random.randrange(d)
    secilenler.append(ad)
    toplam += veriler.values[n, ad]
    
print(toplam)

plt.hist(secilenler)
plt.show