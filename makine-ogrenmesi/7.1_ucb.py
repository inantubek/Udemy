# -*- coding: utf-8 -*-
"""
Created on Wed May 30 16:21:39 2018

@author: IS96347
"""

import pandas as pd
import math
import matplotlib.pyplot as plt

veriler = pd.read_csv("7.1.1_Ads_CTR_Optimisation.csv")

N, d = veriler.shape

tiklamalar = [0] * d
oduller = [0] * d
secilenler = []
toplam = 0

for n in range(0, N):
    max_ucb = 0
    ad = 0
    for i in range(0, d):
        if tiklamalar[i] > 0:
            ort = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2 * math.log(n)/tiklamalar[i])
            ucb = ort + delta
        else:
            ucb = N * d
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    secilenler.append(ad)
    tiklamalar[ad] += 1
    oduller[ad] += veriler.values[n, ad]
    toplam += veriler.values[n, ad]

print(toplam)

plt.hist(secilenler)
plt.show