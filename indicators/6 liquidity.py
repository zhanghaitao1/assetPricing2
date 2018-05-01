# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-22  15:54
# NAME:assetPricing2-6 liquidity.py

import numpy as np
import pandas as pd

from data.dataTools import load_data, save
import statsmodels.formula.api as sm
from collections import OrderedDict

from tool import groupby_rolling

from zht.data.gta.api import read_gta
from zht.utils.dateu import freq_end

#TODO: how about filtering the samples?
def get_liquidity():
    # Turnover rate
    df1=read_gta('Liq_Tover_M',index_col=0)
    df1=df1[df1['Status']=='A'] # A=正常交易
    df1=df1[['Stkcd','Trdmnt','ToverOsM','ToverTlM','ToverOsMAvg','ToverTlMAvg']]
    df1.columns=['sid','t','turnover1','turnover2','turnover3','turnover4']
    df1['t']=freq_end(df1['t'],'M')
    df1['sid']=df1['sid'].astype(str)
    df1=df1.set_index(['t','sid'])
    df1=df1.astype(float)

    # Amihud
    df2=read_gta('Liq_Amihud_M',index_col=0)
    df2 = df2[df2['Status'] == 'A']  # A=正常交易
    df2 = df2[['Stkcd', 'Trdmnt', 'ILLIQ_M']]  # 月内日均换手率(流通股数)
    df2.columns = ['sid', 't', 'amihud']
    df2['t'] = freq_end(df2['t'], 'M')
    df2['sid'] = df2['sid'].astype(str)
    df2=df2.set_index(['t','sid'])
    df2=df2.astype(float)

    '''
    roll1,roll2,zeros1 and zeros2 are note proper for portfolio analysis,since there are a lot of
    zeros in sample,which will cause errors in the program.
    '''
    # roll
    # df3=read_gta('Liq_Roll_M',index_col=0)
    # df3 = df3[df3['Status'] == 'A']  # A=正常交易
    # df3 = df3[['Stkcd', 'Trdmnt', 'Roll_M','Roll_Impact_M']]  # 月内日均换手率(流通股数)
    # df3.columns = ['sid', 't', 'roll1','roll2']
    # df3['t'] = freq_end(df3['t'], 'M')
    # df3['sid'] = df3['sid'].astype(str)
    # df3=df3.set_index(['t','sid'])
    # df3=df3.astype(float)

    # # Zeros
    # df4=read_gta('Liq_Zeros_M',index_col=0)
    # df4 = df4[df4['Status'] == 'A']  # A=正常交易
    # df4 = df4[['Stkcd', 'Trdmnt', 'Zeros_M','Zeros_Impact_M']]  # 月内日均换手率(流通股数)
    # df4.columns = ['sid', 't', 'zeros1','zeros2']
    # df4['t'] = freq_end(df4['t'], 'M')
    # df4['sid'] = df4['sid'].astype(str)
    # df4=df4.set_index(['t','sid'])
    # df4=df4.astype(float)

    # Pastor Stambaugh
    df5=read_gta('Liq_PS_M',index_col=0)
    df5 = df5[df5['Status'] == 'A']  # A=正常交易
    df5 = df5[['Stkcd', 'Trdmnt', 'PSos','PStl']]  # 月内日均换手率(流通股数)
    df5.columns = ['sid', 't', 'ps1','ps2']
    df5['t'] = freq_end(df5['t'], 'M')
    df5['sid'] = df5['sid'].astype(str)
    df5=df5.set_index(['t','sid'])
    df5=df5.astype(float)

    # combine them
    x=pd.concat([df[~df.index.duplicated()] for df in [df1,df2,df5]],axis=1)
    x.columns.name='type'

    save(x,'liquidity')



#---------------------------------------calulate by myself------------------------------
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
    df['t']=freq_end(df['t'], 'D')
    df=df.set_index(['t','sid'])
    if not df.index.is_monotonic_increasing:
        df=df.sort_index(level='t')#TODO: gta's data is not monotonic_increasing ,add this two row to other scripts

    dict=OrderedDict({'1M':15,'3M':50,'6M':100,'12M':200})

    result=groupby_rolling(df,'illiq',dict,_amihud)
    result.index.names=['type','t']

    ln_result=np.log(result)
    ln_result=ln_result.reset_index()
    ln_result['type']='ln_'+ln_result['type'].astype(str)
    ln_result=ln_result.set_index(['type','t'])
    illiq=pd.concat([result,ln_result],axis=0)
    #TODO:use valid observation for the whole project as page 276

    # adjust the format of the DataFrame
    illiq.columns = pd.Index(illiq.columns.astype(str), illiq.columns.name)
    illiq = illiq.reset_index()
    illiq['t'] = freq_end(illiq['t'], 'M')
    illiq = illiq.set_index(['type', 't'])
    illiq = illiq.stack().unstack(level='type')

    #TODO: The data is really noisy,refer to outliers figures for details
    save(illiq,'illiq')

def get_liquidity_ps():
    df=read_gta('Liq_PSM_M')
    #MarketType==21   综合A股和创业板
    # 流通市值加权，but on the page 310,Bali use total market capilization
    condition1=(df['MarketType']==21)
    condition2=(df['ST']==1)#delete the ST stocks

    df = df[condition1 & condition2][['Trdmnt','AggPS_os']]
    df.columns=['t','rm']
    df=df.set_index('t')

    df.index=freq_end(df.index, 'M')
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

    ret=load_data('stockRetM')
    rf=load_data('rfM')
    eret = ret.sub(rf['rf'], axis=0)
    eret = eret.stack()
    eret.index.names=['t','sid']
    eret.name='eret'

    ff3=load_data('ff3M')
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

    result=comb.groupby('sid').apply(_get_result).unstack('sid')

    save(result, 'liqBeta')


if __name__ == '__main__':
    get_liquidity()
