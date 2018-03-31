# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-21  11:17
# NAME:assetPricing2-2beta.py


from dout import *
import statsmodels.formula.api as sm
from tool import monthly_cal


def _get_comb():
    #page 123
    eretD = read_df('eretD', freq='D')
    eretD = eretD.stack()
    eretD.index.names = ['t', 'sid']
    eretD.name = 'eret'
    rpD=read_df('rpD','D')
    combD = eretD.to_frame().join(rpD)

    eretM = read_df('eretM', freq='M')
    eretM = eretM.stack()
    eretM.index.names = ['t', 'sid']
    eretM.name = 'eret'
    rpM=read_df('rpM','M')
    combM = eretM.to_frame().join(rpM)
    return combD,combM

def _beta(subx):
    beta=sm.ols('eret ~ rp',data=subx).fit().params['rp']
    return beta

def cal_beta():
    dictD = {'1M': 15, '3M': 50, '6M': 100, '12M': 200, '24M': 450}
    dictM = {'12M': 10, '24M': 20, '36M': 24, '60M': 24}
    combD,combM=_get_comb()
    monthly_cal(combD,'D',dictD,_beta,'betaD')
    monthly_cal(combM,'M',dictM,_beta,'betaM')


if __name__=='__main__':
    cal_beta()
