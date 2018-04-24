# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-22  16:40
# NAME:assetPricing2-8 idiosyncraticVolatility.py

import pandas as pd
import numpy as np
from functools import partial


from data.dataTools import load_data, save_to_filtered, save
import statsmodels.formula.api as sm
from collections import OrderedDict
from tool import groupby_rolling


def _get_comb():
    eretD=load_data('stockEretD')
    eretD = eretD.stack()
    eretD.index.names = ['t', 'sid']
    eretD.name = 'ret'
    ff3D=load_data('ff3D')
    mktD=load_data('mktRetD')
    mktD.columns=['mkt']
    combD = eretD.to_frame().join(ff3D)
    combD=combD.join(mktD)

    eretM=load_data('stockEretM')
    eretM = eretM.stack()
    eretM.index.names = ['t', 'sid']
    eretM.name = 'ret'
    ffcM=load_data('ffcM')
    mktM=load_data('mktRetM')
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
    
    dfDs=[groupby_rolling(combD,'D',dictD,partial(func,square_m=252**0.5))
          for func in [_vol,_volss,_idioVol_capm,_idioVol_ff3]]

    dfMs=[groupby_rolling(combM,'M',dictM,partial(func,square_m=12**0.5))
          for func in [_vol,_volss,_idioVol_capm,_idioVol_ff3,_idioVol_ffc]]

    for freq,dfs in zip(['D','M'],[dfDs,dfMs]):
        x = pd.concat([df.stack().unstack(level=0) for df in dfs], axis=1)
        x.index.names = ['t', 'sid']
        x.columns.name = 'type'
        save(x,'idio'+freq)

if __name__=='__main__':
    cal_volatility()


