# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-13  09:13
# NAME:assetPricing2-playingField.py
from multiprocessing.pool import Pool

from core.constructFactor import get_single_sorting_assets
from core.ff5 import ts_panel, model_performance
from data.dataApi import Database, Benchmark
from data.dataTools import read_unfiltered
from data.din import parse_financial_report, toMonthly
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


def _read_indicator(name):
    return pd.read_pickle(os.path.join(dirFI,name+'.pkl'))

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

    indicators=corr.index.tolist()
    raise NotImplementedError



def _get_reduced_indicators():
    '''
    only keep one indicator in each category
    Returns:

    '''
    indicators = [el[:-4] for el in os.listdir(dirSpread)]
    g = group_with(indicators, lambda x: x.split('__')[1][:5])
    cn_indicators = [v[0] for v in g.values()]
    return cn_indicators


def get_significant_indicators(bench=None):
    indicators = _get_reduced_indicators()

    ts = []
    for ind in indicators:
        s = pd.read_pickle(os.path.join(dirSpread, ind + '.pkl'))
        t = get_riskAdjusted_alpha_tvalue(s, bench)
        ts.append(t)
    tvalues = pd.Series(ts, index=indicators)
    tvalues = tvalues.sort_values()
    sig_values = tvalues[abs(tvalues) > 1.66]
    return sig_values

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

def get_all_10assets_and_spread_of_databaseIndicator():
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

# selected indicators from database
DATABASE_INDICATORS=['liquidity__turnover1',
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
    '''
    sort independently
    Args:
        indicator:

    Returns:

    '''
    v2 = 'size__size'
    assets=get_25assets(indicator,v2)
    assets.to_pickle(os.path.join(dir25assets,'{}.pkl'.format(indicator)))
    print(indicator)

def get_all_25assets():
    multi_processing(_save_25assets, DATABASE_INDICATORS, pool_size=5)


