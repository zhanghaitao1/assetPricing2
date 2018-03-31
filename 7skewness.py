# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-21  18:17
# NAME:assetPricing2-7skewness.py
from dout import *
import statsmodels.formula.api as sm
from tool import monthly_cal


def _get_comb():
    #TODO:use eretD
    retD=read_df('eretD','D')
    retD=retD.stack()
    retD.index.names=['t','sid']
    retD.name='ret'

    eretD = read_df('stockRetD', freq='D')
    eretD = eretD.stack()
    eretD.index.names = ['t', 'sid']
    eretD.name = 'eret'

    ff3D = read_df('ff3D', 'D')
    mktD = read_df('mktRetD', 'D')
    mktD.columns=['mkt']
    mktD['mkt_square']=mktD['mkt']**2
    retD.head()
    eretD.head()
    multi_comb_D=pd.concat([eretD,retD],axis=1)
    single_comb_D=pd.concat([mktD,ff3D],axis=1)
    combD=multi_comb_D.join(single_comb_D)

    #monthly
    retM = read_df('eretM', 'M')
    retM = retM.stack()
    retM.index.names = ['t', 'sid']
    retM.name = 'ret'

    eretM = read_df('stockRetM', freq='M')
    eretM = eretM.stack()
    eretM.index.names = ['t', 'sid']
    eretM.name = 'eret'

    ff3M = read_df('ff3M', 'M')
    mktM = read_df('mktRetM', 'M')
    mktM.columns = ['mkt']
    mktM['mkt_square'] = mktM['mkt'] ** 2
    retM.head()
    eretM.head()
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
    dictD = {'1M': 15, '3M': 50, '6M': 100, '12M': 200, '24M': 450}
    dictM = {'12M': 10, '24M': 20, '36M': 24, '60M': 24}

    combD,combM=_get_comb()

    monthly_cal(combD, 'D', dictD, _skew, 'skewD')
    monthly_cal(combD, 'D', dictD, _coskew, 'coskewD')
    monthly_cal(combD, 'D', dictD, _idioskew, 'idioskewD')

    monthly_cal(combM, 'M', dictM, _skew, 'skewM')
    monthly_cal(combM, 'M', dictM, _coskew, 'coskewM')
    monthly_cal(combM, 'M', dictM, _idioskew, 'idioskewM')




if __name__=='__main__':
    cal_skewnewss()


