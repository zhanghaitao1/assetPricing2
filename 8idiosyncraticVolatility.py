# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-22  14:50
# NAME:assetPricing2-8idiosyncraticVolatility.py
from collections import OrderedDict
from functools import partial

from dout import *
import numpy as np
import statsmodels.formula.api as sm
from tool import monthly_cal


def _get_comb():
    eretD = read_df('eretD', freq='D')
    eretD = eretD.stack()
    eretD.index.names = ['t', 'sid']
    eretD.name = 'ret'
    ff3D = read_df('ff3D', 'D')
    mktD = read_df('mktRetD', 'D')
    mktD.columns=['mkt']
    combD = eretD.to_frame().join(ff3D)
    combD=combD.join(mktD)

    eretM = read_df('eretM', freq='M')
    eretM = eretM.stack()
    eretM.index.names = ['t', 'sid']
    eretM.name = 'ret'
    ffcM = read_df('ffcM', freq='M')
    mktM = read_df('mktRetM', 'M')
    mktM.columns=['mkt']
    combM = eretM.to_frame().join(ffcM)
    combM=combM.join(mktM)
    return combD,combM


def _vol(df,square_m):
    return df['ret'].std()*100*square_m

def _volss(df,square_m):
    return 100*((np.sum(df['ret']**2)/df.shape[0])**0.5)*square_m

def _idioVol_capm(df,square_m):
    resid=sm.ols('ret ~ rp',data=df).fit().resid
    return ((np.sum(resid**2)/(resid.shape[0]-2))**0.5)*100*square_m

def _idioVol_ff3(df,square_m):
    resid = sm.ols('ret ~ rp + smb + hml', data=df).fit().resid
    return ((np.sum(resid ** 2) / (resid.shape[0] - 4)) ** 0.5)*100*square_m

def _idioVol_ffc(df,square_m):
    resid = sm.ols('ret ~ rp + smb + hml + mom', data=df).fit().resid
    return ((np.sum(resid ** 2) / (resid.shape[0] - 5)) ** 0.5)*100*square_m

def cal_volatility():
    dictD = OrderedDict({'1M': 15, '3M': 50, '6M': 100, '12M': 200, '24M': 450})
    dictM = OrderedDict({'12M': 10, '24M': 20, '36M': 24, '60M': 24})

    combD,combM=_get_comb()

    monthly_cal(combD, 'D', dictD, partial(_vol,square_m=252**0.5), 'volD')
    monthly_cal(combD, 'D', dictD, partial(_volss,square_m=252**0.5), 'volssD')
    monthly_cal(combD, 'D', dictD, partial(_idioVol_capm,square_m=252**0.5), 'idioVol_capmD')
    monthly_cal(combD,'D',dictD,partial(_idioVol_ff3,square_m=252**0.5),'idioVol_ff3D')

    monthly_cal(combM, 'M', dictM, partial(_vol,square_m=12**0.5), 'volM')
    monthly_cal(combM, 'M', dictM, partial(_volss,square_m=12**0.5), 'volssM')
    monthly_cal(combM, 'M', dictM, partial(_idioVol_capm,square_m=12**0.5), 'idioVol_capmM')
    monthly_cal(combM, 'M', dictM, partial(_idioVol_ff3,square_m=12**0.5), 'idioVol_ff3M')
    monthly_cal(combM, 'M', dictM, partial(_idioVol_ffc,square_m=12**0.5), 'idioVol_ffcM')


if __name__=='__main__':
    cal_volatility()



