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
from tool import assign_port_id, my_average, newey_west, multi_processing
from zht.data.gta.api import read_gta
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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



def _read_indicator(name):
    return pd.read_pickle(os.path.join(dirFI,name+'.pkl'))

#parse all the financial indicators
def _filter_indicators(lst):
    newlst=[]
    mark=[]
    for ele in lst:
        if ele[-1].isdigit():
            newlst.append(ele)
        else:
            if ele[:-1] not in mark:
                newlst.append(ele)
                mark.append(ele[:-1])
    return newlst

def parse_all_financial_indicators():
    tbnames=['FI_T{}'.format(i) for i in range(1,12)]
    for tbname in tbnames:
        df=read_gta(tbname)
        varnames=[col for col in df.columns if col not in
                  ['Accper','Indcd','Stkcd','Typrep']]

        if 'Typrep' in df.columns:
            consolidated=True
        else:
            consolidated=False

        varnames=_filter_indicators(varnames)
        for varname in varnames:
            df=parse_financial_report(tbname,varname,consolidated=consolidated)
            df=toMonthly(df)
            df.to_pickle(os.path.join(dirFI,'{}__{}.pkl'.format(tbname,varname)))
            print(tbname,varname)

def indicatorDf_to_10_assets(indicatorDf, indicatorName):
    sampleControl = False
    q = 10

    # data lagged
    s = indicatorDf.stack()
    s.name = indicatorName
    weight = Database(sample_control=sampleControl).by_indicators(['weight'])
    datalagged = pd.concat([s, weight], axis=1)
    datalagged = datalagged.groupby('sid').shift(1)

    # data t
    datat = Database(sample_control=sampleControl).by_indicators(['stockEretM'])
    comb = pd.concat([datalagged, datat], axis=1)
    comb = comb.dropna()

    comb['g'] = comb.groupby('t', group_keys=False).apply(
        lambda df: assign_port_id(df[indicatorName], q))

    assets = comb.groupby(['t', 'g']).apply(
        lambda df: my_average(df, 'stockEretM', wname='weight')) \
        .unstack(level=['g'])
    return assets

def _task(indicator):
    try:
        df=_read_indicator(indicator)
        assets=indicatorDf_to_10_assets(df, indicator)
        assets.to_pickle(os.path.join(dir10assets,'{}.pkl'.format(indicator)))
    except:
        with open(os.path.join(dirProj, 'failed.txt'), 'a') as f:
            f.write(indicator+'\n')
    print(indicator)

def multi_indicator_to_10_assets():
    indicators=[ind[:-4] for ind in os.listdir(dirFI)]
    p=Pool(6)
    p.map(_task,indicators)

def get_assetsSpread(indicator):
    assets=pd.read_pickle(os.path.join(dir10assets,'{}.pkl'.format(indicator)))
    spread=assets[assets.columns.max()]-assets[1]
    spread.name=indicator
    return spread

def get_all_spread():
    indicators=[ind[:-4] for ind in os.listdir(dir10assets)]
    for indicator in indicators:
        spread=get_assetsSpread(indicator)
        spread.to_pickle(os.path.join(dirSpread, '{}.pkl'.format(indicator)))
        print(indicator)

def analyse_corr():
    '''
    analyse the correlation between different factors constructed by soritng
    financial indicators.Since there may be some indicators share the same value.

    Returns:

    '''
    fns=os.listdir(dirSpread)

    ss=[pd.read_pickle(os.path.join(dirSpread, fn)) for fn in fns]
    comb=pd.concat(ss,axis=1)
    corr=comb.corr()
    cc=corr.corr()
    tri=corr.mask(np.triu(np.ones(corr.shape),k=0).astype(bool))
    tri=tri.stack().sort_values(ascending=False)
    tri=tri.reset_index()
    thresh=0.9

def plot_all_spread():
    fns = os.listdir(dirSpread)

    ss = [pd.read_pickle(os.path.join(dirSpread, fn)) for fn in fns]
    comb = pd.concat(ss, axis=1)

    tup=[]
    for col,s in comb.items():
        sharpe_abs=abs(s.mean()/s.std())
        tup.append((col,sharpe_abs))

    tup=sorted(tup,key=lambda x:x[1],reverse=True)

    for i,ele in enumerate(tup):
        indicator=ele[0]
        s=comb[indicator]
        s=s.dropna()
        fig=plt.figure()
        plt.plot(s.index,s.cumsum())
        fig.savefig(os.path.join(os.path.join(dirSpreadFig,indicator + '.png')))
        print(i)

    sp=pd.DataFrame(tup,columns=['indicator','sharpe'])
    sp.to_pickle(os.path.join(dirProj,'sharpe.pkl'))

def select_a_model(i=0):
    sharpe=pd.read_pickle(os.path.join(dirProj, 'sharpe.pkl'))
    indicator=sharpe['indicator'][i]
    factor=pd.read_pickle(os.path.join(dirSpread, indicator + '.pkl'))

    ff3=read_unfiltered('ff3M')
    model=pd.concat([ff3[['rp','smb']],factor],axis=1)
    model=model.dropna()
    return model

def get_10assets_and_spread_for_all_databaseIndicator():
    q=10
    info = Database().info
    indicators = [ele for l in info.values() for ele in l]

    for indicator in indicators:
        assets = get_single_sorting_assets(indicator, q=q)
        assets.to_pickle(os.path.join(dirDatabaseAssets, '{}.pkl'.format(indicator)))
        spread = assets['g{}'.format(q)] - assets['g1']
        spread.name = indicator
        spread.to_pickle(
            os.path.join(dirDatabaseSpread, '{}.pkl'.format(indicator)))
        print(indicator)

def plot_all_databaseSpread():
    fns = os.listdir(dirDatabaseSpread)

    ss = [pd.read_pickle(os.path.join(dirDatabaseSpread, fn)) for fn in fns]
    comb = pd.concat(ss, axis=1)

    tup=[]
    for col,s in comb.items():
        sharpe_abs=abs(s.mean()/s.std())
        tup.append((col,sharpe_abs))

    tup=sorted(tup,key=lambda x:x[1],reverse=True)

    for i,ele in enumerate(tup):
        indicator=ele[0]
        s=comb[indicator]
        s=s.dropna()
        fig=plt.figure()
        plt.plot(s.index,s.cumsum())
        fig.savefig(os.path.join(os.path.join(dirDatabaseSpreadFig,indicator + '.png')))
        print(i)

    sp=pd.DataFrame(tup,columns=['indicator','sharpe'])
    sp=sp.set_index('indicator')['sharpe']
    sp.to_pickle(os.path.join(dirProj,'sharpe_databaseSpread.pkl'))

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

def get_sign(indicator):
    sharpe=pd.read_pickle(os.path.join(dirProj,'sharpe_databaseSpread.pkl'))
    return (1,-1)[sharpe[indicator]<0]

def get_25assets(v1, v2):
    sampleControl = False
    q = 5

    ss=[]
    for v in [v1,v2]:
        if v in Database(sample_control=sampleControl).all_indicators:
            s=Database(sample_control=sampleControl).by_indicators([v])
        else:
            s=pd.read_pickle(os.path.join(dirFI,v+'.pkl')).stack()
            s.name=v
        ss.append(s)

    # data lagged
    weight = Database(sample_control=sampleControl).by_indicators(['weight'])
    datalagged = pd.concat(ss+[weight], axis=1)
    datalagged = datalagged.groupby('sid').shift(1)

    # data t
    datat = Database(sample_control=sampleControl).by_indicators(['stockEretM'])
    comb = pd.concat([datalagged, datat], axis=1)
    comb = comb.dropna()

    comb['g1'] = comb.groupby('t', group_keys=False).apply(
        lambda df: assign_port_id(df[v1], q))
    comb['g2'] = comb.groupby('t', group_keys=False).apply(
        lambda df: assign_port_id(df[v2], q))

    assets = comb.groupby(['t', 'g1', 'g2']).apply(
        lambda df: my_average(df, 'stockEretM', wname='weight'))\
        .unstack(level=['g1','g2'])
    return assets

def _save_25assets(indicator):
    v2 = 'size__size'
    assets=get_25assets(indicator,v2)
    assets.to_pickle(os.path.join(dir25assets,'{}.pkl'.format(indicator)))
    print(indicator)

def get_all_25assets():
    multi_processing(_save_25assets,database_indicators,pool_size=5)

def _riskAdjust(s,bench=None):
    s.name = 'y'
    s=s.to_frame()
    if bench is not None:
        df=pd.concat([s,bench],axis=1)
        formula='y ~ {}'.format(' + '.join(bench.columns.tolist()))
        nw=newey_west(formula,df)
        return nw['Intercept']['t']
    else:
        formula='y ~ 1'
        nw = newey_west(formula, s)
        return nw['Intercept']['t']

def _compare_models_based_on_spread():
    # hxz is best
    ts=[]
    for indicator in database_indicators:
        s=pd.read_pickle(os.path.join(dirDatabaseSpread,indicator+'.pkl'))
        s=s*get_sign(indicator)
        mymodel=select_a_model()
        bs=list(Benchmark().info.keys())
        names=['pure','my']+bs
        benchs=[None,mymodel]+[Benchmark().by_benchmark(r) for r in bs]

        t=pd.Series([_riskAdjust(s,bench) for bench in benchs],index=names)
        ts.append(t)
        print(indicator)

    tvalues=pd.concat(ts, axis=1, keys=database_indicators)
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
    spreadInterceptTvalues=_compare_models_based_on_spread()
    compareWithIntercept10,compareWithJointTest10=compare_models_based_on_assets(assetType='10')
    compareWithIntercept25,compareWithJointTest25=compare_models_based_on_assets(assetType='25')

    spreadInterceptTvalues.to_csv(os.path.join(dirCompare,'spreadInterceptTvalues.csv'))
    compareWithIntercept10.to_csv(os.path.join(dirCompare,'compareWithIntercept10.csv'))
    compareWithJointTest10.to_csv(os.path.join(dirCompare,'compareWithJointTest10.csv'))
    compareWithIntercept25.to_csv(os.path.join(dirCompare,'compareWithIntercept25.csv'))
    compareWithJointTest25.to_csv(os.path.join(dirCompare,'compareWithJointTest25.csv'))


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
        t = pd.Series([_riskAdjust(s, bench) for bench in benchs], index=names)
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






