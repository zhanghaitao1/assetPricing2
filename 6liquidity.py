# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-21  09:04
# NAME:assetPricing2-6liquidity.py

import pandas as pd
import os
import statsmodels.formula.api as sm
import numpy as np
from dout import read_df
from config import DATA_PATH
from tool import groupby_rolling

from zht.data.gta.api import read_gta
from zht.utils.dateu import convert_freq

def _amihud(subx):
    subx['volume']=subx['volume'].replace(0,np.nan)
    subx=subx.dropna()
    #the unit of ret is decimal,that is 0.01 represent 1%,and for volume
    #the unit if Yuan,we multiply the result with 1 billion
    subx['x']=subx['ret'].abs()*1000000000/subx['volume']
    return subx['x'].mean()

def get_amihud_illiq():
    df=read_gta('TRD_Dalyr')
    df=df[['Stkcd','Trddt','Dretwd','Dnvaltrd']]
    df.columns=['sid','t','ret','volume']
    df['t']=convert_freq(df['t'],'D')
    df=df.set_index(['t','sid'])
    dict={'1M':15,'3M':50,'6M':100,'12M':200}

    result=groupby_rolling(df,'illiq',dict,_amihud)
    result.index.names=['type','t']

    ln_result=np.log(result)
    ln_result=ln_result.reset_index()
    ln_result['type']='ln_'+ln_result['type']
    ln_result=ln_result.set_index(['type','t'])
    illiq=pd.concat([result,ln_result],axis=0)
    #TODO:use valid observation for the whole project as page 276
    illiq.to_csv(os.path.join(DATA_PATH,'illiq.csv'))

def get_liquidity_ps():
    df=read_gta('Liq_PSM_M')
    #MarketType==21   综合A股和创业板
    # 流通市值加权，but on the page 310,Bali use total market capilization
    condition1=(df['MarketType']==21)
    condition2=(df['ST']==1)#delete the ST stocks

    df = df[condition1 & condition2][['Trdmnt','AggPS_os']]
    df.columns=['t','rm']
    df=df.set_index('t')

    df.index=convert_freq(df.index,'M')
    df=df.sort_index()
    df['rm_ahead']=df['rm'].shift(1)
    df['delta_rm']=df['rm']-df['rm'].shift(1)
    df['delta_rm_ahead']=df['rm_ahead']-df['rm_ahead'].shift(1)
    #df.groupby(lambda x:x.year).apply(lambda df:df.shape[0])
    #TODO: we don't know the length of window to regress.In this place,we use the five years history
    def regr(df):
        if df.shape[0]>30:
            return sm.ols(formula='delta_rm ~ delta_rm_ahead + rm_ahead',data=df).fit().resid[0]
        else:
            return np.NaN

    window=60 # not exact 5 years
    lm=pd.Series([regr(df.loc[:month][-window:].dropna()) for month in df.index],index=df.index)
    lm.name='lm'

    ret = read_df('stockRetM', freq='M')
    rf = read_df('rfM', freq='M')
    eret = ret.sub(rf['rf'], axis=0)
    eret = eret.stack()
    eret.index.names=['t','sid']
    eret.name='eret'

    ff3=read_df('ff3_gta','M')
    factors=pd.concat([ff3,lm],axis=1)

    comb=eret.to_frame().join(factors)

    def _for_one_month(df):
        if df.shape[0] >=30:
            return sm.ols(formula='eret ~ rp + smb + hml + lm', data=df).fit().params['lm']
        else:
            return np.NaN

    def _get_result(df):
        thresh=30#30 month
        if df.shape[0]>thresh:
            values=[]
            sid=df.index[0][1]
            df = df.reset_index(level='sid', drop=True)
            months=df.index.tolist()[thresh:]
            for month in months:
                subdf=df.loc[:month][-60:]
                subdf=subdf.dropna()
                # df=df.reset_index(level='sid',drop=True).loc[:month].last(window)
                values.append(_for_one_month(subdf))
            print(sid)
            return pd.Series(values,index=months)

    result=comb.groupby('sid').apply(_get_result)
    result.unstack('sid').to_csv(os.path.join(DATA_PATH,'liqBeta.csv'))

#
# if __name__=='__main__':
#     get_liquidity_ps()
#

if __name__=='__main__':
    get_amihud_illiq()