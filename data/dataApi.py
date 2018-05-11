# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-24  21:19
# NAME:assetPricing2-dataApi.py
import os
import pickle

from config import PKL_UNFILTERED_PATH
from data.dataTools import load_data, save, save_to_filtered, read_unfiltered, \
    read_filtered
import pandas as pd
from data.sampleControl import apply_condition

def combine_all_indicators():
    fns=['size','beta','value','momentum','reversal','liquidity',
         'skewness','idio','op','inv','roe']

    xs=[]
    info={}
    for fn in fns:
        x=load_data(fn)
        # stack those panel with only one indicators such as reversal
        if not isinstance(x.index,pd.MultiIndex):
            if x.columns.name=='sid':
                x = x.stack().to_frame()
                x.columns = [fn]

        x.columns = pd.Index(['{}__{}'.format(fn, col) for col in x.columns], x.columns.name)
        xs.append(x)
        info[fn]=x.columns.tolist()

    indicators=pd.concat(xs,axis=1)
    return indicators,info

def combine_all_benchmarks():
    models=['capmM', 'ff3M', 'ffcM', 'ff5M', 'hxz4M','ff6M']
    xs=[]
    info={}
    for model in models:
        x=load_data(model)
        if x.ndim==1:# such as capmM
            x.name='{}__{}'.format(model,x.name)
        else:
            x.columns=pd.Index(['{}__{}'.format(model,col) for col in x.columns],name=x.columns.name)
        xs.append(x)

        if x.ndim==1: # model with only one column such as capmM
            info[model]=[x.name]
        else:
            info[model]=x.columns.tolist()
    benchmark=pd.concat(xs,axis=1)
    return benchmark,info

def join_all():
    '''
        We use the indicators,and weight in time t to predict the adjusted return in time t+1,so,
    for time T we have:
        1. weight
        2. indicators

    For time T+1:
        1. stock excess return
        2. rp
        3. benchmark

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

    # --------------------time T-1 (Backward) ---------------------------------
    weight=load_data('size')['mktCap']
    weight.name='weight'
    indicators,info=combine_all_indicators()

    # -----------------------------time T--------------------------------------
    stockEretM=load_data('stockEretM')
    stockEretM=stockEretM.stack()
    stockEretM.name='stockEretM'

    rfM=load_data('rfM')
    mktRetM=load_data('mktRetM')
    rpM=load_data('rpM')

    #combine singleIndexedr
    single=pd.concat([rfM,mktRetM,rpM],axis=1)

    #combine multiIndexed
    multi=pd.concat([weight,indicators,stockEretM],axis=1)
    data=multi.join(single,how='outer')
    data.index.name=['t','sid']
    data.columns.name='type'

    pickle.dump(info, open(os.path.join(PKL_UNFILTERED_PATH, 'info.pkl'), 'wb'))

    # save info as df
    infoDf=pd.concat([pd.Series(v,name=k) for k,v in info.items()],axis=1)
    infoDf.to_csv('info.csv')

    save(data,'data')

def refine_data():
    data=read_unfiltered('data')
    # data=refine(data) #TODO: filter out the abnormal values
    # save_to_filtered(data,'data')

    data_controlled=apply_condition(data)
    save_to_filtered(data_controlled,'data_controlled')

class Benchmark:
    def __init__(self):
        self.data,self.info=combine_all_benchmarks()

    def by_benchmark(self,name):
        '''

        :param name:one of ['capm','ff3','ff5','ffc','hxz4']
        :return:
        '''
        if name.endswith('M'):
            return self.data[self.info[name]].dropna()
        else:
            return self.data[self.info[name+'M']].dropna()

class Database:
    def __init__(self,sample_control=True):
        if sample_control:
            self.data=read_filtered('data_controlled')
        else:
            self.data=read_unfiltered('data')
        self.info=load_data('info')
        self.all_indicators=[ele for l in self.info.values() for ele in l]

    def by_factor(self,factorname):
        return self.data[self.info[factorname]].copy(deep=True).dropna(how='all')

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
    refine_data()




#TODO:deal with sample control and detect outliers


