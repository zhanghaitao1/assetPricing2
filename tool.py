# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-22  15:03
# NAME:assetPricing2-tool.py

from dout import *
import numpy as np

from zht.utils import assetPricing
import time

#TODO: multiIndex span on the index and groupby


def my_rolling(func, x, window, freq, min_periods, *args, **kwargs):
    '''
    x can be series or Dataframe

    :param func:function running on a series and return a scalar
    :param x:
    :param window:
    :param freq:
    :param min_periods:
    :param args:
    :param kwargs:
    :return:
    '''
    def _rolling_1d(func,s,window,freq,min_periods,*args,**kwargs):
        '''
        s can be singleIndex series or multiIndex Series
        '''
        if isinstance(s.index,pd.MultiIndex):
            indnames=list(s.index.names)
            indnames.remove('t')
            s=s.reset_index(indnames,drop=True)

        if freq == 'M':
            days = s.index.get_level_values('t').unique()
            months = days[days.is_month_end]

            values = []
            for month in months:
                subx = s.loc[:month].last(window).dropna()
                if subx.shape[0] > min_periods:
                    values.append(func(subx, *args, **kwargs))
                else:
                    values.append(np.nan)
            return pd.Series(values, index=months)
        else:
            print('This function only support freq=="M" model')

    if x.ndim==1:
        return _rolling_1d(func,x,window,freq,min_periods,*args,**kwargs)
    elif x.ndim==2:
        return x.apply(lambda s:_rolling_1d(func,s,window,freq,min_periods,*args,**kwargs))



def group_rolling(func,x,groupby,window,freq,min_periods,*args,**kwargs):
    '''
    x must be a Series or DataFrame with multiIndex

    :param func:function running on a series and return a scalar
    :param x:
    :param groupon:
    :param freq:
    :param min_periods:
    :param args:
    :param kwargs:
    :return:
    '''
    def _group_rolling_1d(func, Series, groupby, window,freq,min_periods, *args, **kwargs):

        result=Series.groupby(groupby).apply(
            lambda s:my_rolling(func, s, window, freq, min_periods, *args, **kwargs)
        )
        return result

    if x.ndim==1:
        return _group_rolling_1d(func, x, groupby, window,freq,min_periods, *args, **kwargs)
    elif x.ndim==2:
        return x.apply(lambda s:_group_rolling_1d(func, s, groupby, window,freq,min_periods, *args, **kwargs))


#TODO: use s.rolling to upgrade this fuction as 4momentum.py
def _rolling_for_series(x, months, history, thresh, type_func):
    '''
    calculate the indicator for one stock,and get a time series

    :param x:series or pandas DataFrame
    :param months:list,contain the months to calculate the indicators
    :param history:history length,such as 12M
    :param thresh:the mimium required observe number
    :param type_func:the function name from one of [_skew,_coskew,_idioskew]
    :return:time series
    '''
    # sid=x.index.get_level_values('sid')[0]
    x=x.reset_index('sid',drop=True)
    values=[]
    for month in months:
        #TODO: we use calendar month rather than the absolute 30 days' window
        subx=x.loc[:month].last(history)
        subx=subx.dropna()
        if subx.shape[0]>thresh:
            values.append(type_func(subx))
        else:
            values.append(np.nan)
    return pd.Series(values,index=months)

def groupby_rolling(multiIndDF, prefix, dict, type_func):
    values = []
    names = []
    #TODO:why not use map or other higher-order function

    for history, thresh in dict.items():
        days = multiIndDF.index.get_level_values('t').unique()
        months=pd.date_range(start=days[0],end=days[-1],freq='M')
        value = multiIndDF.groupby('sid').apply(
            lambda df: _rolling_for_series(df, months, history, thresh, type_func))
        values.append(value.T)
        names.append(prefix + '_' + history)
    result = pd.concat(values, axis=0, keys=names)
    return result

#TODO: use closures and decorator to handle this problem
#TODO:upgrade this funcntion use rolling,rolling_apply
def monthly_cal(comb, prefix, dict, type_func, fn):
    result=groupby_rolling(comb,prefix,dict,type_func)
    result.to_csv(os.path.join(DATA_PATH,fn+'.csv'))
    return result

def apply_col_by_col(func):
    '''
        A decorator to convert a function acting on series to a function acting
    on DataFrame,the augmented function will run column by column on the DataFrame

    :param func:
    :return:
    '''
    def augmented(x,*args,**kwargs):
        if x.ndim == 1:
            return func(x,*args,**kwargs)
        elif x.ndim == 2:
            return x.apply(lambda s:func(s,*args,**kwargs)) #TODO: how about apply row by row?

    return augmented


def grouping(x,q,labels,axis=0,thresh=None):
    '''
    sort and name for series or dataframe,for dataframe,axis is required with 0 denoting row-by-row
    and 1 denoting col-by-col
    :param x:
    :param q:
    :param labels:
    :param axis:
    :param thresh:
    :return:
    '''

    def _grouping_1d(series,q,labels,thresh=None):
        if thresh==None:
            thresh=q*10 #TODO:
        series=series.dropna()
        if series.shape[0]>thresh:
            return pd.qcut(series,q,labels)

    if x.ndim==1:
        return _grouping_1d(x,q,labels,thresh)
    else:
        if axis==1:
            return x.apply(lambda s:_grouping_1d(s,q,labels,thresh))
        elif axis==0:
            return x.T.apply(lambda s:_grouping_1d(s,q,labels,thresh))


def assign_port_id(s, q, labels, thresh=None):
    '''
    this function will first dropna and then asign porfolio id.

    :param s: Series
    :param q:
    :param labels:
    :param thresh:
    :return:
    '''
    ns = s.dropna()
    if thresh is None:
        thresh = q * 10  # TODO: thresh self.q*10？

    if ns.shape[0] > thresh:
        result = pd.qcut(ns, q, labels)
        return result
    else:
        return pd.Series(index=ns.index)

def monitor(func):

    def wrapper(*args,**kwargs):
        print('{}   starting -> {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'),func.__name__))
        func(*args,**kwargs)

    return wrapper


def my_average(df,vname,wname=None):
    '''
    calculate average,allow np.nan in df
    This function intensify the np.average by allowing np.nan

    :param df:DataFrame
    :param vname:col name of the target value
    :param wname:col name of the weights
    :return:scalar
    '''
    if wname is None:
        return df[vname].mean()
    else:
        df=df.dropna(subset=[vname,wname])
        if df.shape[0]>0:
            return np.average(df[vname],weights=df[wname])


