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
from config import FILTERED_PATH
from data.base import MyError
from data.dataTools import read_raw, detect_freq, load_data, save_to_filter
from data.outlier import detect_outliers
from pandas.tseries.offsets import MonthEnd
from zht.utils.dateu import freq_end
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

########################################### filtered ##################################################

def filter_stockRetD():
    raw=read_raw('stockRetD')
    x=apply_condition(raw)
    x[abs(x)>0.11]=np.nan

    save_to_filter(x,'stockRetD')

def filter_stockCloseD():
    name='stockCloseD'
    raw=read_raw(name)
    x=apply_condition(raw)
    save_to_filter(x,name)

def filter_mktRetD():
    raw=read_raw('mktRetD')
    x=start_end(raw)
    save_to_filter(x,'mktRetD')

def filter_rfD():
    raw=read_raw('rfD')
    x=start_end(raw)
    save_to_filter(x,'rfD')

def filter_rfM():
    raw = read_raw('rfM')
    x = start_end(raw)
    save_to_filter(x,'rfM')

def filter_stockEretD():
    stockRetD=load_data('stockRetD')
    rfD=load_data('rfD')
    eretD=stockRetD.sub(rfD,axis=0)
    save_to_filter(eretD,'stockEretD')

def filter_stockRetM():
    raw=read_raw('stockRetM')
    x=apply_condition(raw)
    x[abs(x)>1.0]=np.nan

    save_to_filter(x,'stockRetM')

def filter_stockCloseM():
    raw=read_raw('stockCloseM')
    x=apply_condition(raw)
    save_to_filter(x,'stockCloseM')

def filter_stockEretM():
    stockRetM=load_data('stockRetM')
    rfM=load_data('rfM')
    eretM=stockRetM.sub(rfM,axis=0)
    save_to_filter(eretM,'stockEretM')

def filter_mktRetM():
    raw=read_raw('mktRetM')
    x=start_end(raw)
    save_to_filter(x,'mktRetM')

def filter_capM():
    raw=read_raw('capM')
    x=apply_condition(raw)
    save_to_filter(x,'capM')

def filter_bps():
    raw=read_raw('bps')
    x=apply_condition(raw)
    x[abs(x)>100]=np.nan #TODO:

    save_to_filter(x,'bps')

def filter_bps_wind():
    raw=read_raw('bps_wind')
    x=start_end(raw)
    x=x.where(x<100.0)
    save_to_filter(x,'bps_wind')

def filter_stockCloseY():
    raw=read_raw('stockCloseY')
    x=apply_condition(raw)
    detect_outliers(x,'stockCloseY1')
    save_to_filter(x,'stockCloseY')

def filter_ff3M_resset():
    raw=read_raw('ff3M_resset')
    x=start_end(raw)

    save_to_filter(x,'ff3M_resset')

def filter_ff3M():
    raw=read_raw('ff3M')
    x=start_end(raw)
    for col,s in x.iteritems():
        detect_outliers(s,col)

    save_to_filter(x,'ff3M')

def filter_ffcM():
    raw=read_raw('ffcM')
    x=start_end(raw)
    for col,s in x.iteritems():
        detect_outliers(s,'ffcM_'+col)

    save_to_filter(x,'ffcM')

def filter_ff5M():
    raw=read_raw('ff5M')
    x=start_end(raw)

    for col,s in x.iteritems():
        detect_outliers(s,'ff5M_'+col)

    save_to_filter(x,'ff5M')

def filter_hxz4M():
    raw = read_raw('hxz4M')
    x = start_end(raw)

    for col, s in x.iteritems():
        detect_outliers(s, 'hxz4M_' + col)

    save_to_filter(x, 'hxz4M')

def filter_ff3D():
    raw = read_raw('ff3D')
    x = start_end(raw)

    for col, s in x.iteritems():
        detect_outliers(s, 'ff3D_' + col)
    #TODO:noisy
    save_to_filter(x, 'ff3D')

def filter_fpM():
    raw=load_data('ff3M')['rp']
    raw.name='rpM'
    save_to_filter(raw,'rpM')

def filter_rpD():
    raw=load_data('ff3D')['rp']
    raw.name='rpD'
    save_to_filter(raw,'rpD')
















