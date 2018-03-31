# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-22  14:50
# NAME:assetPricing2-8idiosyncraticVolatility.py


from dout import *
import numpy as np
import statsmodels.formula.api as sm
from tool import monthly_cal


def _get_comb():
    #TODO:use eretD rather than eretD
    retD = read_df('stockRetD', freq='D')
    retD = retD.stack()
    retD.index.names = ['t', 'sid']
    retD.name = 'ret'
    ff3D = read_df('ff3D', 'D')
    mktD = read_df('mktRetD', 'D')
    mktD.columns=['mkt']
    combD = retD.to_frame().join(ff3D)
    combD=combD.join(mktD)
    #TODO: use eretM
    retM = read_df('stockRetM', freq='M')
    retM = retM.stack()
    retM.index.names = ['t', 'sid']
    retM.name = 'ret'
    ffcM = read_df('ffc', freq='M')
    mktM = read_df('mktRetM', 'M')
    mktM.columns=['mkt']
    combM = retM.to_frame().join(ffcM)
    combM=combM.join(mktM)
    return combD,combM

def _vol(df):
    return df['ret'].std()*100*252**0.5

def _volss(df):
    return 100*((np.sum(df['ret']**2)/df.shape[0])**0.5)*(252**0.5)

def _idioVol_capm(df):
    resid=sm.ols('ret ~ rp',data=df).fit().resid
    return (np.sum(resid**2)/(resid.shape[0]-2))**0.5

def _idioVol_ff3(df):
    resid = sm.ols('ret ~ rp + smb + hml', data=df).fit().resid
    return (np.sum(resid ** 2) / (resid.shape[0] - 2)) ** 0.5

def _idioVol_ffc(df):
    resid = sm.ols('ret ~ rp + smb + hml + mom', data=df).fit().resid
    return (np.sum(resid ** 2) / (resid.shape[0] - 2)) ** 0.5

def cal_volatility():
    dictD = {'1M': 15, '3M': 50, '6M': 100, '12M': 200, '24M': 450}
    dictM = {'12M': 10, '24M': 20, '36M': 24, '60M': 24}

    combD,combM=_get_comb()

    monthly_cal(combD, 'D', dictD, _vol, 'volD')
    monthly_cal(combD, 'D', dictD, _volss, 'volssD')
    monthly_cal(combD, 'D', dictD, _idioVol_capm, 'idioVol_capmD')
    monthly_cal(combD,'D',dictD,_idioVol_ff3,'idioVol_ff3D')

    monthly_cal(combM, 'M', dictM, _vol, 'volM')
    monthly_cal(combM, 'M', dictM, _volss, 'volssM')
    monthly_cal(combM, 'M', dictM, _idioVol_capm, 'idioVol_capmM')
    monthly_cal(combM, 'M', dictM, _idioVol_ff3, 'idioVol_ff3M')
    monthly_cal(combM, 'M', dictM, _idioVol_ffc, 'idioVol_ffcM')


if __name__=='__main__':
    cal_volatility()



