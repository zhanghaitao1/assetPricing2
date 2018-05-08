# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-07  21:16
# NAME:assetPricing2-10 hxz.py
from core.main import combine_with_datalagged
from data.dataTools import load_data, save
from data.outlier import detect_outliers
from tool import assign_port_id, my_average
import matplotlib.pyplot as plt
import pandas as pd

'''
This replication has some difference with that of Hou, Xue, and Zhang,
 “Digesting Anomalies.”


'''

def get_hxz4():
    v1='size__size'
    v2='inv__inv' #I/A
    v3='op__op' # ROE

    comb = combine_with_datalagged([v1, v2, v3],sample_control=True)
    comb = comb.dropna()

    comb['g1'] = comb.groupby('t', group_keys=False).apply(
        lambda df: assign_port_id(df[v1], 2, range(1, 3)))

    comb['g2'] = comb.groupby(['t', 'g1'], group_keys=False).apply(
        lambda df: assign_port_id(df[v2], [0,0.3,0.7,1.0], range(1, 4)))

    comb['g3'] = comb.groupby(['t', 'g1'], group_keys=False).apply(
        lambda df: assign_port_id(df[v3], [0,0.3,0.7,1.0], range(1, 4)))

    assets = comb.groupby(['t', 'g1', 'g2', 'g3']).apply(
        lambda df: my_average(df, 'stockEretM', wname='weight'))

    df1=assets.groupby(['t','g1']).mean().unstack(level='g1')
    smb=df1[1]-df1[2]

    df2=assets.groupby(['t','g2']).mean().unstack(level='g2')
    ria=df2[3]-df2[1]

    df3=assets.groupby(['t','g3']).mean().unstack(level='g3')
    roe=df3[3]-df2[1]

    rp=load_data('rpM')
    hxz4=pd.concat([rp,smb,ria,roe],axis=1,keys=['rp','smb','ria','roe'])
    hxz4.columns.name='type'
    hxz4=hxz4.dropna()
    save(hxz4,'hxz4M')




#TODO: here,we can only get the data after applying condition on the samples,
#TODO:

'''
1. First layer:base function
2. Second layer:based on First layer
3. application layer call the second layer function

'''


