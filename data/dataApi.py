# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-24  21:19
# NAME:assetPricing2-dataApi.py
from data.dataTools import load_data
import pandas as pd

# fns=['size','beta','momentum','reversal','liquidity','skewness','idiosyncraticVolatility']
fns=['size','momentum','reversal','liquidity']
xs=[]
for fn in fns:
    x=load_data(fn)
    if not isinstance(x.index,pd.MultiIndex):
        x=x.stack().to_frame()
    print(fn,x.columns.name)

    x.columns=pd.Index(['{}__{}'.format(fn,col) for col in x.columns],x.columns.name)

    xs.append(x)

comb=pd.concat(xs,axis=1)





