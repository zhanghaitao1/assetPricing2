# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-06  19:13
# NAME:assetPricing2-data.py
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from config import PROJECT_PATH


from dout import read_df
from pandas.tseries.offsets import MonthEnd

PATH=os.path.join(PROJECT_PATH,'cakici')
THRESH=15 #at list 15 observes for each month.

def get_size():
    #size
    size=read_df('capM','M')
    size=size/1000.0 #TODO:for din convert the size unit.
    size=size.stack()
    size.names='size'
    size.index.names=['t','sid']
    size.to_csv(os.path.join(PATH,'size.csv'))

def get_price():
    #price
    price=read_df('stockRetM','M')
    price=price.stack()
    price.names='price'
    price.index.names=['t','sid']
    price.to_csv(os.path.join(PATH,'price.csv'))

def get_beta():
    #beta
    rf=read_df('rfD','D')
    rm=read_df('mktRetD','D')
    ri=read_df('stockRetD','D')
    df=ri.stack().to_frame()
    df.columns=['ri']
    df=df.join(pd.concat([rf,rm],axis=1))
    df.columns=['ri','rf','rm']
    df.index.names=['t','sid']

    df['y']=df['ri']-df['rf']
    df['x2']=df['rm']-df['rf']
    df['x1']=df.groupby('sid')['x2'].shift(1)

    def _cal_beta(x):
        result=sm.ols('y ~ x1 + x2',data=x).fit().params[['x1','x2']]
        return result.sum()

    def _for_one_sid(x):
        x=x.reset_index('sid')
        sid=x['sid'][0]
        days=x.index.get_level_values('t').unique()
        months=pd.date_range(start=days[0],end=days[-1],freq='M')
        values=[]
        for month in months:
            subx=x.loc[:month].last('1M')
            subx=subx.dropna()
            if subx.shape[0]>THRESH:
                values.append(_cal_beta(subx))
            else:
                values.append(np.nan)
        print(sid)
        return pd.Series(values,index=months)

    beta=df.groupby('sid').apply(_for_one_sid)
    beta.to_csv(os.path.join(PATH,'beta.csv'))

def get_sd():
    #sd
    #TODO: bookmark this function and add it to my repository (pandas handbook),use this method to upgrade the relevant functions
    ri=read_df('stockRetD','D')
    def _rolling_for_one_sid(s):
        ns=s.dropna()
        yearMonth=lambda x:x+MonthEnd(0)
        filtered=ns.groupby(yearMonth).filter(lambda x: x.dropna().shape[0] > THRESH)
        #group by month
        return filtered.groupby(yearMonth).apply(lambda s:s.std())

    sd=ri.apply(_rolling_for_one_sid)
    #TODO: why apply? why not operator on all the columns at one time?


    sd=sd.stack()
    sd.index.names=['t','sid']
    sd.name='sd'

    sd.to_csv(os.path.join(PATH,'sd.csv'))

get_sd()

# speedup
rf = read_df('rfD', 'D')
rm = read_df('mktRetD', 'D')
ri = read_df('stockRetD', 'D')
df = ri.stack().to_frame()
df.columns = ['ri']
df = df.join(pd.concat([rf, rm], axis=1))
df.columns = ['ri', 'rf', 'rm']
df.index.names = ['t', 'sid']

df['y'] = df['ri'] - df['rf']
df['x2'] = df['rm'] - df['rf']
df['x1'] = df.groupby('sid')['x2'].shift(1)
df=df[['y','x1','x2']]
df=df[-400000:]
# operator on multiple columns
data1=df.copy()
##############
# method 1
df1=data1.copy()





# operator on one column (or series).Take calculting std for example
data2=df['y']

#########################
# method1:            unstack() then operator on dataframe
df1=data2.unstack()
#filter
result1=df1.groupby(lambda dt: dt + MonthEnd(0)).apply(lambda x:x.dropna(axis=1, thresh=15).std())
result1=result1.unstack()
result1.to_csv(r'e:\a\std1.csv')

######################
# method 2: groupby and then apply on series
df2=data2.copy()

def _one_sid(s):
    # s is multiIndex DataFrame
    s=s.reset_index('sid',drop=True)
    _get_monthend=lambda dt:dt+MonthEnd(0)
    ns=s.groupby(_get_monthend).filter(lambda x:x.dropna().shape[0]>15)
    result=ns.groupby(_get_monthend).apply(lambda x:x.std())
    return result

result2=df2.groupby('sid').apply(_one_sid)
result2.index.names=['sid','t']
result2=result2.reorder_levels(['t','sid'])
result2.sort_index(level='t').head()
result2=result2.unstack()
result2.to_csv(r'e:\a\std2.csv')

##########################################
# method 3:
df3=data2.unstack()
def _one_col(s):
    # s is singleIndex dataframe
    _get_monthend=lambda dt:dt+MonthEnd(0)
    ns=s.groupby(_get_monthend).filter(lambda x:x.dropna().shape[0]>15)
    result=ns.groupby(_get_monthend).apply(lambda x:x.std())
    return result

result3=df3.apply(_one_col)
result3.to_csv(r'e:\a\std3.csv')




#see
def get_see():
    see=pd.read_csv(os.path.join(PROJECT_PATH,r'data\idioskewD.csv'),index_col=[0,1],parse_dates=True)
    see.index.names=['type','t']
    see=see.reset_index()
    see=see[see['type']=='D_1M']
    see=see.drop(columns=['type'])
    see=see.set_index('t')
    see=see.stack()
    see.name='see'
    see.index.names=['t','sid']
    see.to_csv(os.path.join(PATH,'see.csv'))

def get_strev():
    #strev
    strev=pd.read_csv(os.path.join(PROJECT_PATH,r'data\reversal.csv'),index_col=[0,1],parse_dates=True)
    strev.columns=['strev']
    strev.to_csv(os.path.join(PATH,'strev.csv'))

#mom
def get_mom():
    mom=pd.read_csv(os.path.join(PROJECT_PATH,r'data\momentum.csv'),index_col=[0,1],parse_dates=True)
    mom=mom[['mom']]
    mom.to_csv(os.path.join(PATH,'mom.csv'))

def get_bkmt():
    # monthly
    bkmt=read_df('bm','M')
    bkmt=bkmt.stack()
    bkmt.name='bkmt'
    bkmt.index.names=['t','sid']
    bkmt.to_csv(os.path.join(PATH,'bkmt.csv'))

#---------------------------
def get_new_src():
    p=r'E:\a\75_2612_STK_MKT_Dalyr\STK_MKT_Dalyr.txt'
    df=pd.read_csv(p,sep='\t', encoding='ISO-8859-1', error_bad_lines=False, skiprows=[1, 2])
    df.to_csv(r'E:\a\75_2612_STK_MKT_Dalyr\STK_MKT_Dalyr.csv',encoding='utf-8')


#------------------------------------


def get_cfpr():
    #TODO:
    df=pd.read_csv(r'D:\zht\database\quantDb\sourceData\gta\data\csv\STK_MKT_Dalyr.csv',encoding='gbk')
    # PCF is 市现率＝股票市值/去年经营现金流量净额
    df=df[['TradingDate','Symbol','PCF']]
    df.columns=['t','sid','pcf']
    df['t']=pd.to_datetime(df['t'])
    df['sid']=df['sid'].astype(str)
    df['pcf'][df['pcf']<=1]=np.nan #TODO:
    df['cfpr']=1.0/df['pcf']
    df=df.set_index(['t','sid'])

    def _get_monthEnd(x):
        #TODO: bookmark this function about groupby as_index,group_keys.
        sid=x.index.get_level_values('sid')[0]
        print(sid)
        x=x.reset_index('sid',drop=True)
        # we use df.sort_index.iloc[-1,:] to get a series rather than using df.sort_index()[-1:] to
        # get a dataframe.In this way,lastrow will be a singleIndex dataframe and the index is monthend.
        # If we use df.sort_index()[-1:],lastrow will be a multiIndex DataFrame with level(0) combine from monthend
        # and level(1) coming from df.
        lastrow=x.groupby(lambda t:t+MonthEnd(0)).apply(lambda df:df.sort_index().iloc[-1,:])
        # The src is of poor quality and it contains some out-of-order samples.
        #TODO:observe gta src data.especially the order of date.
        return lastrow

    result=df.groupby('sid').apply(_get_monthEnd)
    result.index.names=['sid','t']
    result=result['cfpr']
    result.to_csv(os.path.join(PATH,'cfpr.csv'))

def get_ep():
    df=pd.read_csv(r'D:\zht\database\quantDb\sourceData\gta\data\csv\STK_MKT_Dalyr.csv',encoding='gbk')
    # PE is 市盈率＝股票市总值/最近四个季度的归属母公司的净利润之和
    df=df[['TradingDate','Symbol','PE']]
    df.columns=['t','sid','pe']
    df['t']=pd.to_datetime(df['t'])
    df['sid']=df['sid'].astype(str)
    df[df['pe']<=0]=np.nan
    df['ep']=1.0/df['pe']
    df=df.set_index(['t','sid'])
    df=df[['ep']]
    #------------------
    #TODO:bookmark
    #TODO: usually,it may same much time to unstack(),do something on dataframe and then stack(),
    #TODO: rather than groupby and operator on series,groupby should be used to store panel data,rather than
    #TODO: speedup the calculation.
    #TODO: compare the groupby().apply with apply to test which one is faster.
    df=df.unstack()
    df=df.sort_index()
    df=df.groupby(lambda dt:dt+MonthEnd(0)).apply(lambda x:x.iloc[-1,:])
    df=df.stack()
    df.to_csv(os.path.join(PATH,'ep.csv'))

def get_ret():
    ret=read_df('stockRetM','M')
    ret=ret.stack()
    ret.name='ret'
    ret.index.names=['t','sid']
    ret.to_csv(os.path.join(PATH,'ret.csv'))

def combine_all():
    fns=os.listdir(PATH)
    dfs=[]
    for fn in fns:
        df=pd.read_csv(os.path.join(PATH,fn),index_col=[0,1],parse_dates=True)
        dfs.append(df)

        print(fn,df.shape)
    comb=pd.concat(dfs,axis=1)

    print(comb.shape)


combine_all()



