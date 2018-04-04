# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-04  11:14
# NAME:assetPricing2-test_tool.py

from dout import *
import statsmodels.formula.api as sm
from tool import monthly_cal,group_rolling


def _get_comb():
    '''
    page 321

    :return:
    '''
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


def compare_monthly_cal_with_group_rolling():
    dictD = {'1M': 15, '3M': 50, '6M': 100, '12M': 200, '24M': 450}
    dictM = {'12M': 10, '24M': 20, '36M': 24, '60M': 24}

    combD,combM=_get_comb()

    result=group_rolling(_skew,combD,'sid','1M','M',15)

    print('test')



if __name__=="__main__":
    compare_monthly_cal_with_group_rolling()




