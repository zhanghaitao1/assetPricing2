# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-08  10:57
# NAME:assetPricing2-a.py
from data.dataTools import load_data
import pandas as pd

data=load_data('data')
skew=load_data('skewness')

months=skew.index.get_level_values('t').unique()
month=months[64]

sub1=skew.groupby('t').get_group(months[63]) #1999-7
# sub1.to_csv(r'e:\a\sub1.csv')

retD = load_data('stockRetD')
retD = retD.stack()
retD.index.names = ['t', 'sid']
retD.name = 'ret'

eretD = load_data('stockEretD')
eretD = eretD.stack()
eretD.index.names = ['t', 'sid']
eretD.name = 'eret'

