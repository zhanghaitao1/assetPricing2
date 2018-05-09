# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-06  19:13
# NAME:assetPricing2-get_data.py
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from config import PROJECT_PATH
from data.dataTools import load_data

from dout import read_df
from pandas.tseries.offsets import MonthEnd

PATH=os.path.join(PROJECT_PATH,'cakici')
THRESH=15 #at list 15 observes for each month.
def save(df,name):
    df.to_pickle(os.path.join(PATH,name+'.pkl'))

def handle_data():
    size=load_data('size')['size']
    save(size,'size')

    price=load_data('stockCloseM').stack()
    price.name='price'
    save(price,'price')


    beta=pd.read_csv(os.path.join(PATH,'beta.csv'),
                     index_col=[0,1],parse_dates=True)['beta']
    save(beta,'beta')

    sd=pd.read_csv(os.path.join(PATH,'sd.csv'),
                   index_col=[0,1],parse_dates=True)['sd']
    save(sd,'sd')

    see=pd.read_csv(os.path.join(PATH,'see.csv'),
                   index_col=[0,1],parse_dates=True)['see']
    save(see,'see')

    strev=pd.read_csv(os.path.join(PATH,'strev.csv'),
                   index_col=[0,1],parse_dates=True)['strev']
    save(strev,'strev')

    factors=pd.read_csv(os.path.join(PATH,'factors.csv'),
                       index_col=[0,1],parse_dates=True)
    factors.head()

    fns = list(filter(lambda fn:fn.endswith('.csv'),os.listdir(PATH)))
    dfs = []
    for fn in fns:
        df = pd.read_csv(os.path.join(PATH, fn), index_col=[0, 1], parse_dates=True)
        dfs.append(df)
        print(fn,df.shape)






    comb = pd.concat(dfs, axis=1)
    comb = comb.dropna(axis=0, how='all')
    save(comb,'comb')



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
        # x is multiIndex Dataframe
        nx=x.reset_index('sid')
        sid=nx['sid'][0]
        print(sid)
        _get_monthend=lambda dt:dt+MonthEnd(0)
        #filter out those months with observations less than THRESH
        nx=nx.groupby(_get_monthend).filter(lambda a:a.dropna().shape[0]>=THRESH)
        if nx.shape[0]>0:
            result=nx.groupby(_get_monthend).apply(_cal_beta)
            return result

    beta=df.groupby('sid').apply(_for_one_sid)
    beta.index.names=['sid','t']
    beta=beta.reorder_levels(['t','sid']).sort_index(level='t')
    beta.name='beta'
    beta.to_frame().to_csv(os.path.join(PATH,'beta.csv'))

def get_sd():
    #sd
    #TODO: bookmark this function and add it to my repository (pandas handbook),use this method to upgrade the relevant functions
    ri=read_df('stockRetD','D')*100 #TODO: the unit is percent (%)

    #filter
    '''
    can not use 

    sd=ri.resample('M').std()
    
    directly,since you need to filter out invalid samples before calculate std

    def _cal_std(x):
        if x.notnull().sum()>=THRESH: #TODO: compare with x.dropna().shape[0],which one is faster?
            return x.std()

    sd0=ri.resample('M').agg(_cal_std)

    sometimes,groupby can be more flexible than resample
    '''
    #TODO: use resampling
    _get_monthend = lambda x: x + MonthEnd(0)
    sd=ri.groupby(_get_monthend).apply(lambda df:df.dropna(axis=1,thresh=THRESH).std())

    sd.index.names=['t','sid']
    sd.name='sd'
    sd.to_frame().to_csv(os.path.join(PATH,'sd.csv'))

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
    see.to_frame().to_csv(os.path.join(PATH,'see.csv'))

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
    bkmt.to_frame().to_csv(os.path.join(PATH,'bkmt.csv'))

def get_cfpr(): #TODO: wrong! data available problems,lag 6 months
    df=pd.read_csv(r'D:\zht\database\quantDb\sourceData\gta\data\csv\STK_MKT_Dalyr.csv',encoding='gbk')
    # PCF is 市现率＝股票市值/去年经营现金流量净额
    df=df[['TradingDate','Symbol','PCF']]
    df.columns=['t','sid','pcf']
    df['t']=pd.to_datetime(df['t'])
    df['sid']=df['sid'].astype(str)
    df['pcf'][df['pcf']<=1]=np.nan #TODO:
    df['cfpr']=1.0/df['pcf']
    df=df.set_index(['t','sid'])
    df=df['cfpr']
    df=df.unstack()
    # TODO:observe gta src data.especially the order of date.The src is of poor quality and it contains some out-of-order samples.
    df=df.sort_index()
    df=df.resample('M').agg(lambda x:x[-1])
    df=df.stack()
    df.index.names=['t','sid']
    df.name='cfpr'
    df.to_frame().to_csv(os.path.join(PATH,'cfpr.csv'))

def get_ep(): #TODO: wrong! lag 6 months
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
    df=df.unstack()
    df=df.sort_index()
    df=df.resample('M').agg(lambda x:x[-1])
    # this two method is equaly
    # df2=df.groupby(lambda dt:dt+MonthEnd(0)).apply(lambda x:x.iloc[-1,:])
    df=df.stack()
    df.to_frame().to_csv(os.path.join(PATH,'ep.csv'))

def get_ret():
    ret=read_df('stockRetM','M')
    ret=ret.stack()
    ret.name='ret'
    ret.index.names=['t','sid']
    ret.to_frame().to_csv(os.path.join(PATH,'ret.csv'))

def combine_all():
    fns=os.listdir(PATH)
    dfs=[]
    for fn in fns:
        df=pd.read_csv(os.path.join(PATH,fn),index_col=[0,1],parse_dates=True)
        dfs.append(df)

    comb=pd.concat(dfs,axis=1)
    comb=comb.dropna(axis=0,how='all')
    comb.to_csv(os.path.join(PATH,'factors.csv'))





