# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-22  16:40
# NAME:assetPricing2-8 idiosyncraticVolatility.py

import pandas as pd
import numpy as np
from functools import partial
from multiprocessing.pool import Pool


from data.dataTools import load_data, save_to_filtered, save
import statsmodels.formula.api as sm
from collections import OrderedDict
from tool import groupby_rolling, groupby_rolling1


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

        x.to_pickle(r'e:\a\tmp_idio{}.pkl'+freq)

        # save(x,'idio'+freq)

def task(arg):
    result=groupby_rolling1(*arg)
    print(arg[1].func.__name__,arg[2],arg[3])
    return result


if __name__ == '__main__':
    dictD = OrderedDict({'1M': 15, '3M': 50, '6M': 100, '12M': 200, '24M': 450})
    dictM = OrderedDict({'12M': 10, '24M': 20, '36M': 24, '60M': 24})
    combD,combM=_get_comb()

    p = Pool(5)
    argsD = [(combD, partial(func,square_m=252**0.5), history, thresh) for func in [_vol,_volss,_idioVol_capm,_idioVol_ff3]
             for history, thresh in dictD.items()]
    argsM = [(combM, partial(func,square_m=12**0.5), history, thresh) for func in [_vol,_volss,_idioVol_capm,_idioVol_ff3,_idioVol_ffc]
             for history, thresh in dictM.items()]

    dfDs = p.map(task, argsD)
    dfMs = p.map(task, argsM)

    xs=[]
    for freq,dfs,args in zip(['D','M'],[dfDs,dfMs],[argsD,argsM]):
        x = pd.concat([df.stack() for df in dfs], axis=1,
                      keys=['{}_{}__{}'.format(func.func.__name__[1:],history,freq)
                            for _,func,history,_ in args])
        x=x.reorder_levels(order=['t','sid']).sort_index()
        x.columns.name = 'type'
        xs.append(x)

    save(pd.concat(xs,axis=1),'idio',sort_axis=False)


