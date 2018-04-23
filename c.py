# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-23  20:22
# NAME:assetPricing2-c.py

import pandas as pd


N=1

df1=pd.DataFrame({'A':[1,2,3,4,5,6]*N, 'B':[1,5,3,4,5,7]*N}, index=pd.date_range('20140101 101501', freq='u', periods=6*N))

df2=pd.DataFrame({'D':[10,2,30,4,5,10]*N, 'F':[1,5,3,4,5,70]*N}, index=pd.date_range('20140101 101501.000003', freq='u', periods=6*N))

df1.merge(df2, how='outer', right_index=True, left_index=True).fillna(method='ffill')


df1.align(df2,)

pd.set_option('max_rows',10)

