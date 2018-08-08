# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-22  14:30
# NAME:assetPricing2-check_benchmarks.py
from data.dataApi import Benchmark
import pandas as pd

BM=Benchmark()

dfs=[]
for m in ['capm','ff3','ff5','ffc','hxz4']:
    dfm=BM.by_benchmark(m)
    # dfm.cumsum().plot().get_figure().show()
    dfs.append(dfm)

comb=pd.concat(dfs,axis=1)
comb.cumsum().plot().get_figure().show()

inds=['rp','hml','smb','cma','mom','ria','roe']

for ind in inds:
    cols=[col for col in comb.columns if col.endswith(ind)]
    comb[cols].dropna().cumsum().plot().get_figure().show()


