# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-21  14:55
# NAME:assetPricing2-4momentum.py

from config import *
from dout import read_df
import numpy as np
import pandas as pd


def __cal_cumulative_return(s):
    return np.cumprod(s + 1)[-1] - 1


def _before(s, interval, min_periods):
    '''
    for d_before,do not include the return of time t

    :param s:
    :param interval:
    :param min_periods:
    :return:
    '''
    # In this place,s is an ndarray,it equals np.array(series.dropna()),
    # we are not sure s[:-1] has deleted which value,it may be value of time t,and it can also
    # be the value of time t-2
    return s.rolling(interval, min_periods=min_periods).apply(lambda s: __cal_cumulative_return(s[:-1]))

def _upto(s, interval, min_periods):
    return s.rolling(interval, min_periods=min_periods).apply(__cal_cumulative_return)

def get_momentum():
    stockRetM=read_df('stockRetM','M')
    stk=stockRetM.stack()
    stk.index.names=['t','sid']
    #lagged 1 month
    d_lag={'mom':[12,9],#since the window is 11,and we do not use the value of time t,so,here we set 12 rather than 11
           'r12':[13,10],
           'r6':[7,5]}
    #nonlagged
    d_nonlag={'R12M':[12,10],
                'R9M':[9,7],
                'R6M':[6,5],
                'R3M':[3,3]}
    ss=[]
    names=[]
    for bn,bp in d_lag.items():
        ser=stk.groupby('sid').apply(lambda s:_before(s,bp[0],bp[1]))
        ss.append(ser)
        names.append(bn)

    for un,up in d_nonlag.items():
        ser=stk.groupby('sid').apply(lambda s:_upto(s,up[0],up[1]))
        ss.append(ser)
        names.append(un)

    momentum=pd.concat(ss,axis=1,keys=names)
    momentum=momentum*100
    momentum.to_csv(os.path.join(DATA_PATH,'momentum.csv'))
    # for col in momentum.columns:
    #     momentum[col].unstack().to_csv(os.path.join(DATA_PATH,col+'.csv'))

if __name__=='__main__':
    get_momentum()

