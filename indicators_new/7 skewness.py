# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-22  16:00
# NAME:assetPricing2-7 skewness.py

import pandas as pd

from data.dataTools import load_data, save_to_filter
import statsmodels.formula.api as sm
from collections import OrderedDict
from tool import groupby_rolling

def _get_comb():
    '''
    page 321

    :return:
    '''
    retD=load_data('stockRetD')
    retD=retD.stack()
    retD.index.names=['t','sid']
    retD.name='ret'

    eretD=load_data('stockEretD')
    eretD = eretD.stack()
    eretD.index.names = ['t', 'sid']
    eretD.name = 'eret'

    ff3D=load_data('ff3D')
    mktD=load_data('mktRetD').to_frame()
    mktD.columns=['mkt']
    mktD['mkt_square']=mktD['mkt']**2
    multi_comb_D=pd.concat([eretD,retD],axis=1)
    single_comb_D=pd.concat([mktD,ff3D],axis=1)
    combD=multi_comb_D.join(single_comb_D)

    #monthly
    retM=load_data('stockRetM')
    retM = retM.stack()
    retM.index.names = ['t', 'sid']
    retM.name = 'ret'

    eretM=load_data('stockEretM')
    eretM = eretM.stack()
    eretM.index.names = ['t', 'sid']
    eretM.name = 'eret'

    ff3M=load_data('ff3M')
    mktM=load_data('mktRetM').to_frame()
    mktM.columns = ['mkt']
    mktM['mkt_square'] = mktM['mkt'] ** 2
    multi_comb_M = pd.concat([eretM, retM], axis=1)
    single_comb_M = pd.concat([mktM, ff3M], axis=1)
    combM = multi_comb_M.join(single_comb_M)
    return combD,combM

def _skew(subx):
    return subx['ret'].skew()

def _coskew(subx):
    #TODO: eret ,rp rather than
    coskew=sm.ols('eret ~ mkt + mkt_square',data=subx).fit().params['mkt_square']
    return coskew

def _idioskew(subx):
    resids = sm.ols('eret ~ rp + smb + hml', data=subx).fit().resid
    idioskew = resids.skew()
    return idioskew


def cal_skewnewss():
    dictD = OrderedDict({'1M': 15, '3M': 50, '6M': 100, '12M': 200, '24M': 450})
    dictM = OrderedDict({'12M': 10, '24M': 20, '36M': 24, '60M': 24})

    combD,combM=_get_comb()

    skewD=groupby_rolling(combD, 'D', dictD, _skew)
    coskewD=groupby_rolling(combD, 'D', dictD, _coskew)
    idioskewD=groupby_rolling(combD, 'D', dictD, _idioskew)

    skewM=groupby_rolling(combM, 'M', dictM, _skew)
    coskewM=groupby_rolling(combM, 'M', dictM, _coskew)
    idioskewM=groupby_rolling(combM, 'M', dictM, _idioskew)

    save_to_filter(skewD,'skewD')
    save_to_filter(coskewD,'coskewD')
    save_to_filter(idioskewD,'idioskewD')
    save_to_filter(skewM,'skewM')
    save_to_filter(coskewM,'coskewM')
    save_to_filter(idioskewM,'idioskewM')



if __name__=='__main__':
    cal_skewnewss()


