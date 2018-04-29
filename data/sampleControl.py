# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-21  17:40
# NAME:assetPricing2-sampleControl.py
import datetime

import numpy as np
import pandas as pd
from data.base import MyError
from data.dataTools import read_unfiltered, detect_freq, load_data, save_to_filtered
from data.outlier import detect_outliers, delete_outliers
from pandas.tseries.offsets import MonthEnd
from zht.utils.mathu import get_inter_frame

def control_sid(conditions):
    '''
    is_sz
    is_sh
    is_gem 创业板
    is_cross
    not_financial
    is_industry

    :param conditions:
    :return:stock code
    '''
    #TODO: is_gem,is_industry,
    condition_set=['is_sz','is_sh','not_cross','not_financial']
    info=read_unfiltered('listInfo')

    def _one_condition(condition):
        if condition in condition_set:
            sids=info.index[info[condition]]
            return sids
        else:
            raise ValueError('The "conditions" should be one of {}'.format(repr(condition_set)))
    if isinstance(conditions, list):
        l_sids=[_one_condition(con) for con in conditions]
        return sorted(list(set.intersection(*map(set,l_sids))))

def control_t(start='1997-01-01', end=None, freq='M'):
    '''
    The limit on return starts from 1996-12-26

    start
    end

    is_bear
    is_bull
    is_downside?

    :return:
    '''
    '''
    use truncating  & fancy indexing
    refer to http://pandas.pydata.org/pandas-docs/stable/timeseries.html#

    '''
    if not end:
        end=datetime.datetime.today()

    return pd.date_range(start,end,freq=freq)

def cross_closePrice_floor(clsPrice=5.0,freq='M'):
    '''
    delete penny stocks

    the minimum close price is 5

    :param clsPrice:
    :param freq:
    :return:
    '''
    stockClose=read_unfiltered('stockClose' + freq)
    return stockClose>clsPrice

#TODO: refer to readme.md to find more controling methods.
def cross_year_after_list(freq='M'):
    '''
    listed at list 1 year
    :return:
    '''
    listInfo=read_unfiltered('listInfo')
    listInfo['year_later']=listInfo['listDate']+pd.offsets.DateOffset(years=1)
    if freq=='M':
        listInfo['year_later']=listInfo['year_later']+MonthEnd(1)
        # 1 rather than 0,exclude the first month,since most of
        # year_later won't be monthend.
    else:
        listInfo['year_later']=listInfo['year_later']+pd.offsets.DateOffset(days=1)

    mask=listInfo[['year_later']].copy()
    mask.columns=['t']
    mask['bool']=True
    mask=mask.reset_index().set_index(['t','sid'])['bool']
    mask=mask.unstack()
    mask=mask.reindex(index=pd.Index(pd.date_range(mask.index[0],mask.index[-1],freq=freq),name=mask.index.name))
    mask=mask.ffill()
    return mask

def cross_not_st(freq='M'):
    if freq=='M':
        stInfo=read_unfiltered('stInfoM')
    elif freq=='D':
        stInfo=read_unfiltered('stInfoD')
    else:
        raise MyError('freq must belong to ["M","D"] rather than {}'.format(freq))
    return stInfo


def control_input(freq):
    sids=control_sid(['not_financial'])
    t=control_t(start='2001-01-01',freq=freq)
    cross1=cross_closePrice_floor(freq=freq)
    cross2=cross_year_after_list(freq=freq)
    cross3=cross_not_st(freq=freq)
    cross1,cross2,cross3=get_inter_frame([cross1,cross2,cross3])
    comb=cross1 & cross2 & cross3
    comb=comb.reindex(index=pd.Index(t,name='t'),columns=pd.Index(sids,name='sid'))
    comb=comb.dropna(axis=0,how='all')
    comb=comb.dropna(axis=1,how='all')
    return comb

def apply_condition(x):
    '''
    combine all types of sample controling methods
    :param x:
    :return:
    '''
    freq=detect_freq(x.index)
    condition=control_input(freq)
    if isinstance(x.index,pd.MultiIndex):
        return x.loc[x.index.intersection(condition.stack().dropna().index)]
    else:
        x,condition=get_inter_frame([x,condition])
        return x[condition.fillna(value=False)]





#TODO: sid add suffix


#TODO:The 10% limit policy took effect at the beginning of 1997.We exclude stocks that have been listed for less than one year and returns on the first day after the initial public offering.
#TODO: as robustness check in Long, Jiang, and Zhu, “Idiosyncratic Tail Risk and Expected Stock Returns: Evidence from the Chinese Stock Markets.”
'''
Delete ST (special treatment) and firms in financial industry.
    How to delete ST,delete from the ST date or delete the ST stocks from the whole sample?

susamples,before and after December 2006.
subsamples with returns higher and lower than the median index return
subsamples before and after March 2010 when Chinese stock markets partially allowed short sales.
calculate portfolio returns with different holding epriods of 2,6,and 12 months
'''





#-------------------------------old --------------------------------------
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
    info=read_unfiltered('listInfo')

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
    stockCloseM=read_unfiltered('stockClose' + freq)
    stockCloseM,df=get_inter_frame([stockCloseM,df])
    return df[stockCloseM>=clsPrice]

def sample_data_optimization():
    pass

def roof_price():
    pass

def in_event_window():
    pass


#TODO: refer to readme.md to find more controling methods.
def year_after_list(df):
    '''
    listed at list 1 year
    :return:
    '''
    freq=detect_freq(df.index)

    listInfo=read_unfiltered('listInfo')
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
        stInfo=read_unfiltered('stInfoM')
    elif freq=='D':
        stInfo=read_unfiltered('stInfoD')
    else:
        raise MyError('freq must belong to ["M","D"] rather than {}'.format(freq))
    df,stInfo=get_inter_frame([df,stInfo])

    return df[stInfo.notnull()]

def bear_or_bull():
    # refer to marketStates.py
    pass



########################################### filtered ##################################################


