# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-22  14:57
# NAME:assetPricing2-1 beta.py
import os
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd

from data.dataTools import load_data, save_to_filtered, save
import statsmodels.formula.api as sm
from collections import OrderedDict
from tool import groupby_rolling, groupby_rolling1


def _get_comb():
    #page 123
    combs=[]
    for freq in ['D','M']:
        eret=load_data('stockEret'+freq).stack()
        eret.name='eret'
        rp=load_data('rp'+freq)
        rp.name='rp'
        comb=eret.to_frame().join(rp)
        combs.append(comb)
    return tuple(combs)


def _beta(subx):#TODO: use scipy or numpy rather than sm.ols
    beta=sm.ols('eret ~ rp',data=subx).fit().params['rp']
    return beta

def cal_beta():
    dictD = OrderedDict({'1M': 15, '3M': 50, '6M': 100, '12M': 200, '24M': 450})#TODO: why so many months are lost? refer to betaD.csv
    dictM = OrderedDict({'12M': 10, '24M': 20, '36M': 24, '60M': 24})
    combD,combM=_get_comb()

    betaD=groupby_rolling(combD,'D',dictD,_beta)
    betaM=groupby_rolling(combM,'M',dictM,_beta)

    betaD=betaD.stack().unstack(level=0)
    betaM=betaM.stack().unstack(level=0)

    #combine
    x = pd.concat([betaD, betaM], axis=1)
    x.index.names = ['t', 'sid']
    x.columns.name = 'type'

    save(x,'beta',sort_axis=False)

if __name__ == '__main__':
    cal_beta()
    #TODO: parallelize

