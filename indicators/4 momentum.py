# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-22  15:49
# NAME:assetPricing2-4 momentum.py

import pandas as pd
import numpy as np


from data.dataTools import load_data, save
import statsmodels.formula.api as sm
from collections import OrderedDict
from tool import groupby_rolling


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
    # we are not sure which value has been deleted by s[:-1],it may be value of time t,and it can also
    # be the value of time t-2
    return s.rolling(interval, min_periods=min_periods).apply(lambda s: __cal_cumulative_return(s[:-1]))

def _upto(s, interval, min_periods):
    return s.rolling(interval, min_periods=min_periods).apply(__cal_cumulative_return)

def get_momentum():
    stockRetM=load_data('stockRetM')
    stk=stockRetM.stack()
    stk.index.names=['t','sid']
    #lagged 1 month
    #Te one month lag is imposed to avoid the short-term reversal eﬀect frst documented by Jegadeesh (1990)
    d_lag=OrderedDict({'mom':[12,9],#since the window is 11,and we do not use the value of time t,so,here we set 12 rather than 11
                    'r12':[13,10],
                    'r6':[7,5]})
    #nonlagged
    d_nonlag=OrderedDict({'R12M':[12,10],
                        'R9M':[9,7],
                        'R6M':[6,5],
                        'R3M':[3,3]})
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
    momentum.columns.name='type'
    momentum=momentum*100
    momentum.columns.name='type'

    save(momentum,'momentum',sort_axis=False)

if __name__=='__main__':
    get_momentum()


