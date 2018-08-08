# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-23  13:49
# NAME:assetPricing2-beta_sw_dm.py
from data.dataTools import read_filtered, save
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import os
import statsmodels.formula.api as sm


MIN_SAMPLES=15 #at list 15 observes for each month.

def get_beta_sw_dm():
    '''
    refer to page 5 of cakici for details about this beta.

    Returns:

    '''
    #beta
    rf=read_filtered('rfD')
    rm=read_filtered('mktRetD')
    ri=read_filtered('stockRetD')
    df=ri.stack().to_frame()
    df.columns=['ri']
    df=df.join(pd.concat([rf,rm],axis=1))
    df.columns=['ri','rf','rm']
    df.index.names=['t','sid']

    df['y']=df['ri']-df['rf']
    df['x2']=df['rm']-df['rf']
    df['x1']=df.groupby('sid')['x2'].shift(1)

    def _cal_beta(x):
        result=sm.ols('y ~ x1 + x2',data=x).fit().params[['x1','x2']]
        return result.sum()

    def _for_one_sid(x):
        # x is multiIndex Dataframe
        nx=x.reset_index('sid')
        sid=nx['sid'][0]
        print(sid)
        _get_monthend=lambda dt:dt+MonthEnd(0)
        #filter out those months with observations less than MIN_SAMPLES
        nx=nx.groupby(_get_monthend).filter(lambda a: a.dropna().shape[0] >= MIN_SAMPLES)
        if nx.shape[0]>0:
            result=nx.groupby(_get_monthend).apply(_cal_beta)
            return result

    beta=df.groupby('sid').apply(_for_one_sid)
    beta.index.names=['sid','t']
    beta=beta.reorder_levels(['t','sid']).sort_index(level='t')
    beta.name='beta'

    save(beta,'beta_sw_dm')

if __name__ == '__main__':
    get_beta_sw_dm()
