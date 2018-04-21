# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-21  17:40
# NAME:assetPricing2-sampleControl.py
import datetime
import os

import numpy as np
import pandas as pd
from data.base import MyError
from data.check import check
from data.dataTools import read_raw, detect_freq
from data.outlier import detect_outliers
from pandas.tseries.offsets import MonthEnd
from zht.utils.mathu import get_inter_frame


def control_stock_sample(df, condition):
    '''
    is_sz
    is_sh
    is_gem 创业板
    is_cross
    not_financial
    is_industry

    :param df:
    :param condition:
    :return:stock code
    '''
    #TODO: is_gem,is_industry,
    conditions=['is_sz','is_sh','not_cross','not_financial']
    info=read_raw('listInfo')

    if condition in conditions:
        validSids=info.index[info[condition]]
    else:
        raise ValueError('The "condition" should be one of {}'.format(repr(conditions)))

    if df.index.name=='sid':
        return df.reindex(index=df.index.intersection(validSids)) #TODO: intersection rather than reindex
    elif df.columns.name=='sid':
        return df.reindex(columns=df.columns.intersection(validSids))
    else:
        raise MyError('No axis named "sid"!')


def start_end(df, start='1997-01-01', end=None):
    '''
    The limit on return starts from 1996-12-26

    start
    end

    is_bear
    is_bull
    is_downside?

    :return:
    '''
    if isinstance(start,str):
        start=pd.to_datetime(start)

    if not end:
        end=datetime.datetime.today()

    if isinstance(end,str):
        end=pd.to_datetime(end)

    if 't' in df.index.names:
        return df[(start<=df.index.get_level_values('t')) &(df.index.get_level_values('t')<=end)]
    elif 't' in df.columns:
        return df[(start<=df['t'])&(df['t']<=end)]
    else:
        raise ValueError('There is no index or column named "t" !')

def floor_price(df,clsPrice=5.0):
    '''
    delete penny stocks

    the minimum close price is 5

    :param df:
    :param clsPrice:
    :return:
    '''
    freq=detect_freq(df.index)
    stockCloseM=read_raw('stockClose'+freq)
    stockCloseM,df=get_inter_frame([stockCloseM,df])
    return df[stockCloseM>=clsPrice]

def sample_data_optimization():
    pass

def roof_price(df,price):
    pass

def in_event_window(df):
    pass


#TODO: refer to readme.md to find more controling methods.
def year_after_list(df):
    '''
    listed at list 1 year
    :return:
    '''
    freq=detect_freq(df.index)

    listInfo=read_raw('listInfo')
    listInfo['year_later']=listInfo['listDate']+pd.offsets.DateOffset(years=1)
    if freq=='M':
        listInfo['year_later']=listInfo['year_later']+MonthEnd(1)
        # 1 rather than 0,exclude the first month,since most of
        # year_later won't be monthend.
    else:
        listInfo['year_later']=listInfo['year_later']+pd.offsets.DateOffset(days=1)

    start=listInfo['year_later'].min()
    end=datetime.datetime.today()
    mark=pd.DataFrame(np.nan,index=pd.date_range(start,end,freq=freq),
                       columns=listInfo.index)

    for sid,d in listInfo['year_later'].iteritems():
       mark.at[d,sid]=1

    mark=mark.ffill()
    df,mark=get_inter_frame([df,mark])
    df=df[mark.notnull()].dropna(axis=1,how='all')
    return df

def delete_st(df):
    freq=detect_freq(df.index)
    if freq=='M':
        stInfo=read_raw('stInfoM')
    elif freq=='D':
        stInfo=read_raw('stInfoD')
    else:
        raise MyError('freq must belong to ["M","D"] rather than {}'.format(freq))
    df,stInfo=get_inter_frame([df,stInfo])

    return df[stInfo.notnull()]


def apply_condition(df):
    df=control_stock_sample(df,'not_financial')
    df=start_end(df)
    df=floor_price(df)
    df=year_after_list(df)
    df=delete_st(df)
    return df


df=read_raw('stockRetD')
detect_outliers(df,'0')
newdf=apply_condition(df)
detect_outliers(newdf,'1')

