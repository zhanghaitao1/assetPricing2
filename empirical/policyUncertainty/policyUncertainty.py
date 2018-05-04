# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-23  00:22
# NAME:assetPricing2-policyUncertainty.py

import os
from collections import OrderedDict

import pandas as pd
from data.dataTools import load_data, save
from pandas.tseries.offsets import MonthEnd
from tool import groupby_rolling
from zht.utils.dateu import freq_end
import pylab as plt
import statsmodels.formula.api as sm


def visulize_pu():
    pu=load_data('pu')
    pu.plot()
    plt.show()

def _get_comb():
    eretM=load_data('eretM')
    eretM=eretM.stack()
    eretM.index.names=['t','sid']
    eretM.name='eret'
    pu=load_data('pu')
    combM=eretM.to_frame().join(pu)
    return combM

def _sensitivity_to_pu(subx):
    sen=sm.ols('eret ~ pu',data=subx).fit().params['pu']
    return sen

def cal_sen():
    dictM=OrderedDict({'12M':10,'24M':20,'36M':24,'60M':24})
    combM=_get_comb()
    sen=groupby_rolling(combM,'M',dictM,_sensitivity_to_pu)
    sen=sen*100
    sen=sen.stack().unstack(level=0)
    sen.index.names=['t','sid']
    sen.columns.name='type'
    save(sen,'sen',outliers=False)
cal_sen()


#TODO: check the outliers calculated by myself,such as beta,coskewness and so on
#TODO: unify the format of beta ,skew and so on
#TODO: how to check outliers of multiIndexed DataFrame,take sen for example.


