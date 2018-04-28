# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-24  21:19
# NAME:assetPricing2-dataApi.py
import os
import pickle

from config import PKL_PATH
from data.dataTools import load_data, save
import pandas as pd
from data.sampleControl import apply_condition


def combine_all_indicators():
    fns=['size','beta','value','momentum','reversal','liquidity','skewness','idio']

    xs=[]
    info_s=[]
    for fn in fns:
        x=load_data(fn)
        # stack those panel with only one indicators such as reversal
        if not isinstance(x.index,pd.MultiIndex):
            if x.columns.name=='sid':
                x = x.stack().to_frame()
                x.name = fn

        x.columns = pd.Index(['{}__{}'.format(fn, col) for col in x.columns], x.columns.name)
        xs.append(x)
        info_s.append(pd.Series(x.columns,name=fn))

    indicators=pd.concat(xs,axis=1)
    info=pd.concat(info_s,axis=1)
    pickle.dump(info,open(os.path.join(PKL_PATH,'info.csv.pkl'),'wb'))
    info.to_csv('info.csv.csv')
    return indicators

def combine_all_benchmarks():
    models=['capmM', 'ff3M', 'ffcM', 'ff5M', 'hxz4M']
    xs=[]
    for bench in models:
        x=load_data(bench)
        if x.ndim==1:# such as capmM
            x.name='{}__{}'.format(bench,x.name)
        else:
            x.columns=pd.Index(['{}__{}'.format(bench,col) for col in x.columns],name=x.columns.name)
        xs.append(x)

    benchmark=pd.concat(xs,axis=1)
    return benchmark

def join_all():
    # time T
    weight=load_data('size')['mktCap']
    weight.name='weight'
    indicators=combine_all_indicators()
    indicators=indicators.groupby('sid').shift(1)
    '''
        all the indicators are shift forward one month except for eret,rf and other base data,
    so the index denotes time t+1,and all the indicators are from time t,the base data are from 
    time t+1.We adjust the indicators rather than the base data for these reasons:
    1. we will sort the indicators in time t to construct portfolios and analyse the eret in time
        t+1
    2. We need to make sure that the index for eret and benchmark is corresponding to the time when 
    it was calcualted. If we shift back the base data in this place (rather than shift forward the
    indicators),we would have to shift forward eret again when we regress the portfolio eret on 
    benckmark model in the function _alpha in template.py

    For simply,we use the information at t to predict the eret of time t+1.In our DATA.data,the index
    denotes time t,and the values for eretM,benchmark model and so on is from time t+1.

    Notice:
        To calculate value-weighted result,we use the market capitalization of the time t (portfolio
        formation period) as weight.So,in this place,we should shift the capM forward for one step
        as well.For more details,refer to page 40 of Bali.

    '''
    # time T+1
    stockEretM=load_data('stockEretM')
    stockEretM=stockEretM.stack()
    stockEretM.name='stockEretM'

    rfM=load_data('rfM')
    mktRetM=load_data('mktRetM')
    rpM=load_data('rpM')
    benchmark=combine_all_benchmarks()

    #combine singleIndexed
    single=pd.concat([rfM,mktRetM,rpM,benchmark],axis=1)

    #combine multiIndexed
    multi=pd.concat([indicators,stockEretM],axis=1)
    data=multi.join(single,how='outer')
    data.index.name=['t','sid']
    data.columns.name='type'

    data_controlled=apply_condition(data)
    save(data,'data')
    save(data_controlled,'data_controlled')

class Dataset:
    def __init__(self,sample_control=True):
        if sample_control:
            self.data=load_data('data_controlled')
        else:
            self.data=load_data('data')
        self.info=load_data('info.csv')

    def by_factor(self,factorname):
        return self.data[self.info[factorname].dropna().values].copy(deep=True).dropna(how='all')

    def by_indicators(self,indicators):
        '''
        no mather indicators is just a string represent one indicators
        or list (tuple),the function will return a DataFrame
        :param indicators:
        :return: DataFrame
        '''
        if isinstance(indicators,(list,tuple)):
            return self.data[list(indicators)].copy(deep=True).dropna(how='all')
        else:
            return self.data[[indicators]].copy(deep=True).dropna(how='all')


if __name__ == '__main__':
    join_all()





