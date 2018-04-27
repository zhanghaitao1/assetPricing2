# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-24  21:19
# NAME:assetPricing2-dataApi.py
from data.dataTools import load_data, save
import pandas as pd

fns=['size','beta','value','momentum','reversal','liquidity','skewness','idio','marketStates']

multi_xs = []
single_xs = []
for fn in fns:
    x=load_data(fn)
    # stack those panel with only one indicators such as reversal
    if not isinstance(x.index,pd.MultiIndex):
        if x.columns.name=='sid':
            x = x.stack().to_frame()
            x.name = fn

    x.columns = pd.Index(['{}__{}'.format(fn, col) for col in x.columns], x.columns.name)
    if isinstance(x.index,pd.MultiIndex):
        multi_xs.append(x)
    else:
        single_xs.append(x)

multi=pd.concat(multi_xs,axis=1)
single=pd.concat(single_xs,axis=1)


