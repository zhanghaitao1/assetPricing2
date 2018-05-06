# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-08  10:57
# NAME:assetPricing2-a.py
from data.dataTools import load_data
import pandas as pd

import numpy as np
from scipy.stats import f, norm
import matplotlib.pyplot as plt

# first f
rv1 = f(dfn=3, dfd=15, loc=0, scale=1)
x = np.linspace(rv1.ppf(0.0001), rv1.ppf(0.9999), 100)
y = rv1.pdf(x)
rv1.cdf(3)
rv1.sf(3)
1-rv1.cdf(3)


plt.xlim(0,5)
plt.plot(x,y, 'b-')
plt.show()


