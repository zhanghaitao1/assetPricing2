# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-22  14:57
# NAME:assetPricing2-2 beta.py

import numpy as np
import pandas as pd

from data.dataTools import load_data, save_to_filter
import statsmodels.formula.api as sm
from collections import OrderedDict
from tool import groupby_rolling


def _get_comb():
    #page 123
    eretD=load_data('eretD')
    eretD = eretD.stack()
    eretD.index.names = ['t', 'sid']
    eretD.name = 'eret'
    rpD=load_data('rpD')
    rpD.name='rp'
    combD = eretD.to_frame().join(rpD)
    
    eretM=load_data('eretM')
    eretM = eretM.stack()
    eretM.index.names = ['t', 'sid']
    eretM.name = 'eret'
    rpM=load_data('rpM')
    rpM.name='rp'
    combM = eretM.to_frame().join(rpM)
    return combD,combM

def _beta(subx):
    beta=sm.ols('eret ~ rp',data=subx).fit().params['rp']
    return beta

def cal_beta():
    #TODO: name D1M D3M  M12M M36M

    dictD = OrderedDict({'1M': 15, '3M': 50, '6M': 100, '12M': 200, '24M': 450})#TODO: why so many months are lost? refer to betaD.csv
    dictM = OrderedDict({'12M': 10, '24M': 20, '36M': 24, '60M': 24})
    combD,combM=_get_comb()
    # combD=combD[-800000:]
    # combM=combM[-40000:]
    betaD=groupby_rolling(combD,'D',dictD,_beta)
    betaM=groupby_rolling(combM,'M',dictM,_beta)
    save_to_filter(betaD,'betaD')
    save_to_filter(betaM,'betaM')


if __name__=='__main__':
    cal_beta()
