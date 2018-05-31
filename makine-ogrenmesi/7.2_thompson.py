# -*- coding: utf-8 -*-
"""
Created on Thu May 31 08:17:38 2018

@author: IS96347
"""

import pandas as pd
import random
import matplotlib.pyplot as plt

veriler = pd.read_csv("7.1.1_Ads_CTR_Optimisation.csv")

N, d = veriler.shape

birler = [0] * d
sifirlar = [0] * d
secilenler = []
toplam = 0
for n in range(0, N):
    max_th = 0
    ad = 0
    for i in range(0, d):
        beta = random.betavariate(birler[i] + 1, sifirlar[i] + 1)
        if beta > max_th:
            max_th = beta
            ad = i
    secilenler.append(ad)
    odul = veriler.values[n, ad]
    if odul == 1:
        birler[ad] += 1
    else:
        sifirlar[ad] += 1
    toplam += odul

print(toplam)

plt.hist(secilenler)
plt.show