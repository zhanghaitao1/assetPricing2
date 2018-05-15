# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-12  08:46
# NAME:assetPricing2-compare.py
from multiprocessing.pool import Pool

from core.constructFactor import get_single_sorting_assets
from core.ff5 import ts_panel, model_performance
from data.dataApi import Database, Benchmark
from data.dataTools import read_unfiltered
from data.din import parse_financial_report, toMonthly
from empirical.my.compareBenchmark.playingField import _get_reduced_indicators, \
    get_significant_indicators
from tool import assign_port_id, my_average, newey_west, multi_processing, \
    get_riskAdjusted_alpha_tvalue
from zht.data.gta.api import read_gta
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zht.utils.listu import group_with

dirProj= r'D:\zht\database\quantDb\researchTopics\assetPricing2\my'
dirFI=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\financial_indicators'
dir10assets=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\10assets'
dirSpread= r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\10minus1'
dirSpreadFig=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\spread_fig'
dirDatabaseAssets=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\databaseAssets'
dirDatabaseSpread=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\database_10Minus1'
dirCompare=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\compare_models'
dir25assets=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\25assets'
dirDatabaseSpreadFig=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\databaseSpread_fig'
dirIndustryIndex=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\industryIndex'


def select_a_model(i=0):
    sharpe=pd.read_pickle(os.path.join(dirProj, 'sharpe.pkl'))
    indicator=sharpe['indicator'][i]
    factor=pd.read_pickle(os.path.join(dirSpread, indicator + '.pkl'))

    ff3=read_unfiltered('ff3M')
    model=pd.concat([ff3[['rp','smb']],factor],axis=1)
    model=model.dropna()
    return model

database_indicators=['liquidity__turnover1',
                     'idio__idioVol_capm_1M__D',
                     'liquidity__amihud',
                     'momentum__R3M',
                     'skewness__skew_24M__D',
                     'reversal__reversal',
                     'value__bm',
                     'inv__inv',
                     'beta__D_1M',
                     'op__op',
                     'roe__roe']

bench=Benchmark().by_benchmark('ff3') #TODO:
sig_indicators=get_significant_indicators(bench)
reduced_indicators=sig_indicators.index.tolist()

def get_sign(indicator):
    if indicator in database_indicators:
        sharpe=pd.read_pickle(os.path.join(dirProj,'sharpe_databaseSpread.pkl'))
        return (1,-1)[sharpe[indicator]<0]
    elif indicator in reduced_indicators:
        return np.sign(sig_indicators[indicator])

def _spread_tvalues():
    # hxz is best
    mymodel = select_a_model()
    bs = list(Benchmark().info.keys())
    names = ['pure', 'my'] + bs
    benchs = [None, mymodel] + [Benchmark().by_benchmark(r) for r in bs]

    ts=[]
    for indicator in database_indicators+reduced_indicators:
        if indicator in database_indicators:
            s=pd.read_pickle(os.path.join(dirDatabaseSpread,indicator+'.pkl'))
        else:
            s=pd.read_pickle(os.path.join(dirSpread,indicator+'.pkl'))

        # s=s*get_sign(indicator) #TODO:
        t=pd.Series([get_riskAdjusted_alpha_tvalue(s, bench) for bench in benchs], index=names)
        ts.append(t)
        print(indicator)

    tvalues=pd.concat(ts, axis=1, keys=database_indicators+reduced_indicators)
    return tvalues

def compare_models_based_on_assets(assetType='25'):
    if assetType=='25':
        directory=dir25assets
    elif assetType=='10':
        directory=dirDatabaseAssets
    else:
        raise ValueError

    byInterceptLst=[]
    byJointTest=[]
    for indicator in database_indicators:
        assets=pd.read_pickle(os.path.join(directory,'{}.pkl'.format(indicator)))
        mymodel = select_a_model()
        bs = list(Benchmark().info.keys())
        benchNames = ['pure', 'my'] + bs
        benchs = [None, mymodel] + [Benchmark().by_benchmark(r) for r in bs]

        interceptResult=[]
        jointTestResult=[]
        for bench in benchs:
            interceptResult.append(ts_panel(assets,bench))
            if bench is not None:
                jointTestResult.append(model_performance(assets.copy(),bench))
        byInterceptLst.append(pd.concat(interceptResult,axis=0,keys=benchNames))
        byJointTest.append(pd.concat(jointTestResult,axis=1,keys=benchNames[1:]).T)
        print(indicator)

    compareWithIntercept=pd.concat(byInterceptLst,axis=0,keys=database_indicators)
    compareWithJointTest=pd.concat(byJointTest,axis=0,keys=database_indicators)
    return compareWithIntercept,compareWithJointTest

def compare():
    spreadInterceptTvalues=_spread_tvalues()
    # compareWithIntercept10,compareWithJointTest10=compare_models_based_on_assets(assetType='10')
    # compareWithIntercept25,compareWithJointTest25=compare_models_based_on_assets(assetType='25')

    spreadInterceptTvalues.to_csv(os.path.join(dirCompare,'spreadInterceptTvalues.csv'))
    # compareWithIntercept10.to_csv(os.path.join(dirCompare,'compareWithIntercept10.csv'))
    # compareWithJointTest10.to_csv(os.path.join(dirCompare,'compareWithJointTest10.csv'))
    # compareWithIntercept25.to_csv(os.path.join(dirCompare,'compareWithIntercept25.csv'))
    # compareWithJointTest25.to_csv(os.path.join(dirCompare,'compareWithJointTest25.csv'))

compare()

def clean_industryIndex():
    '''
    中证全指一级行业指数
    wind 一级行业指数
    wind 二级行业指数
    中信二级行业指数

    Returns:

    '''
    for i in [1, 2, 3, 4]:
        fn = 'industryIndex{}'.format(i)
        industryIndex=pd.read_csv(os.path.join(dirIndustryIndex,fn+'.csv')
                                  ,index_col=0,
                                  encoding='gbk')
        def _convert_date(ind):
            year,month,day=tuple(ind.split('/'))
            month=month if len(month)==2 else '0'+month
            return '-'.join([year,month,day])

        industryIndex.index=industryIndex.index.map(_convert_date)
        industryIndex.index=pd.to_datetime(industryIndex.index)
        # industryIndex.columns=['industry{}'.format(i) for i in range(1,industryIndex.shape[1]+1)]
        industryIndex.to_pickle(os.path.join(dirIndustryIndex,'{}.pkl'.format(fn)))

def compare_models_based_on_industryIndex(fn):
    industryIndex=pd.read_pickle(os.path.join(dirIndustryIndex,'{}.pkl'.format(fn)))
    ts = []
    for name,s in industryIndex.items():
        mymodel = select_a_model()
        bs = list(Benchmark().info.keys())
        names = ['pure', 'my'] + bs
        benchs = [None, mymodel] + [Benchmark().by_benchmark(r) for r in bs]
        t = pd.Series([get_riskAdjusted_alpha_tvalue(s, bench) for bench in benchs], index=names)
        ts.append(t)
        print(name)

    industryIndexTvalues = pd.concat(ts, axis=1, keys=industryIndex.columns)
    return industryIndexTvalues

def compare_models_based_on_industryIndex_assets(fn):
    assets=pd.read_pickle(os.path.join(dirIndustryIndex,'{}.pkl'.format(fn)))
    mymodel = select_a_model()
    bs = list(Benchmark().info.keys())
    benchNames = ['pure', 'my'] + bs
    benchs = [None, mymodel] + [Benchmark().by_benchmark(r) for r in bs]

    jointTestResult = []
    for bench in benchs:
        if bench is not None:
            jointTestResult.append(model_performance(assets.copy(), bench))
    result=pd.concat(jointTestResult, axis=1, keys=benchNames[1:]).T
    return result

def analyse_with_industryIndex():
    clean_industryIndex()
    for i in [1,2,3,4]:
        fn='industryIndex{}'.format(i)
        r1=compare_models_based_on_industryIndex(fn)
        r2=compare_models_based_on_industryIndex_assets(fn)
        r1.to_csv(os.path.join(dirCompare,'compare_with_industryIndexTvalues_{}.csv'.format(i)))
        r2.to_csv(os.path.join(dirCompare,'compare_with_industryJointTest_{}.csv'.format(i)))


'''
ideas:
1. find anomalies (use capm to identify)
2. compare with time series regression,single sorts and bivariate-sorts
3. GRS and so on.
4. explain each other


#TODO: set q=5 and test run again
'''