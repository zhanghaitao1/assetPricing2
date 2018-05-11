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
from data.dataTools import read_unfiltered, detect_freq
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
    :return:a list of stock codes
    '''
    #TODO: is_gem,is_industry,
    condition_set=['is_sz','is_sh','not_cross','not_financial']
    info=read_unfiltered('listInfo')

    def _one_condition(condition):
        if condition in condition_set:
            sids=info[info[condition]].index.tolist()
            return sids
        else:
            raise ValueError('The "conditions" should be one of {}'.format(repr(condition_set)))

    if isinstance(conditions,str):
        return _one_condition(conditions)
    elif isinstance(conditions, list):
        l_sids=[_one_condition(con) for con in conditions]
        return sorted(list(set.intersection(*map(set,l_sids))))
    else:
        raise MyError('no such conditon as {}'.format(conditions))

def control_t(start='1997-01-01', end=None, freq='M'):
    '''
    The limit on return starts from 1996-12-26

    start
    end

    is_bear
    is_bull
    is_downside?

    :return:time series
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
    :return:DataFrame filled with True or False
    '''
    stockClose=read_unfiltered('stockClose' + freq)
    return stockClose>clsPrice

#TODO: refer to readme.md to find more controling methods.
def cross_year_after_list(freq='M'):
    '''
    listed at list 1 year
    :return:DataFrame filled with True or False
    '''
    listInfo=read_unfiltered('listInfo')
    listInfo['year_later']=listInfo['listDate']+pd.offsets.DateOffset(years=1)
    if freq=='M':
        listInfo['year_later']=listInfo['year_later']+MonthEnd(1)
        # 1 rather than 0,exclude the first month,since most of
        # year_later won't be monthend.
    else:
        listInfo['year_later']=listInfo['year_later']+\
                               pd.offsets.DateOffset(days=1)

    mask=listInfo[['year_later']].copy()
    mask.columns=['t']
    mask['bool']=True
    mask=mask.reset_index().set_index(['t','sid'])['bool']
    mask=mask.unstack()
    mask=mask.reindex(index=pd.Index(pd.date_range(
        mask.index[0],mask.index[-1],freq=freq),name=mask.index.name))
    mask=mask.ffill()
    mask=mask.fillna(value=False) # replace nan or None with False
    return mask

def cross_filter_out_less_than_120_records_in_past_12_months():
    '''
    refer to A.1 in Appendix of Liu, Stambaugh, and Yuan, “Size and Value in China.”

    We also impose several filters: First,we exclude stocks that have become
    public within the past six months. Second,we exclude stocks having less
    than 120 days of trading records during the past 12 monbths. We also
    exclude stocks having less than 15 days of trading records during the
    most recent months. The above filters are intended to prevent our results
    from being influenced by returns that follow long trading suspensions.

    Returns:

    '''

    pass


def cross_not_st(freq='M'):
    '''
    filter out st stocks
    :param freq:
    :return: DataFrame filled with True or False
    '''
    if freq=='M':
        stInfo=read_unfiltered('stInfoM')
    elif freq=='D':
        stInfo=read_unfiltered('stInfoD')
    else:
        raise MyError('freq must belong to ["M","D"] rather than {}'.format(freq))
    return stInfo.fillna(value=False) # replace nan or None with False

def cross_size_groups(freq='M'):
    '''
    'all-but-tiny' stocks are those larger than the NYSE 20th percentile and 'large'
    stocks are those larger than the NYSE 50th percentile based on market equity at
    the beginning of the month.Fama and French (2008) suggest usign these groups as
    a simple way to check whether predictability is driven by micro-cap stocks or also
    exists among the economically more important population of large stocks.

    references:
        Lewellen, J. (2015). The Cross-section of Expected Stock Returns. Critical Finance Review 4, 1–44.

    :return:three DataFrames filled with True or False
    '''
    p1=0.3
    p2=0.7
    size=read_unfiltered('capM')
    floors=size.quantile(p1,axis=1)
    roofs=size.quantile(p2,axis=1)

    small=[]
    medium=[]
    big=[]
    for t,s in size.iterrows():
        f=floors[t]
        r=roofs[t]
        small.append(s<f)
        medium.append((f<s) & (f<r))
        big.append(s>=r)

    small=pd.concat(small,axis=1,keys=size.index).T
    medium=pd.concat(medium,axis=1,keys=size.index).T
    big=pd.concat(big,axis=1,keys=size.index).T

    return small,medium,big

#TODO: to control the sample,we have to use the lagged value to decide the sample
#TODO: is valid or not.That is,at time t,we should use the size of time t-1 to
#TODO: decide whethe a stock is a tiny stock or a big stock.??????


def combine_condition(freq):
    '''
    :param freq:
    :return: DataFrame filled with True or False
    '''
    sids=control_sid(['not_financial'])
    t=control_t(start='1997-01-01',freq=freq)
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
    condition=combine_condition(freq)
    if isinstance(x.index,pd.MultiIndex):
        stk=condition.stack()
        interIndex=x.index.intersection(stk.index)
        x=x.reindex(index=interIndex)
        stk=stk.reindex(index=interIndex)
        return x[stk]
    else:
        x,condition=get_inter_frame([x,condition])
        return x[condition]

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
def sample_data_optimization():
    pass

def roof_price():
    pass

def in_event_window():
    pass


#TODO: refer to readme.md to find more controling methods.


def bear_or_bull():
    # refer to marketStates.py
    pass



########################################### filtered ##################################################


