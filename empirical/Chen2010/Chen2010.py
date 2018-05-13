# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-09  16:23
# NAME:assetPricing2-Chen2010.py
from core.constructFactor import single_sorting_factor
from core.ff5 import regression_details_5x5, ts_panel, model_performance
from data.dataApi import Database, Benchmark
from data.dataTools import read_unfiltered
from data.din import parse_financial_report, toMonthly
from tool import assign_port_id, my_average, multi_processing, newey_west
from zht.data.gta.api import read_gta
import os
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
import numpy as np

BENCH=Benchmark()


direc= r'D:\zht\database\quantDb\researchTopics\assetPricing2\data\FI'
dirData= r'D:\zht\database\quantDb\researchTopics\assetPricing2\data\FI\data'
figPath= r'D:\zht\database\quantDb\researchTopics\assetPricing2\data\FI\fig'
factorPath= r'D:\zht\database\quantDb\researchTopics\assetPricing2\data\FI\factor'

dirFactor_database=r'D:\zht\database\quantDb\researchTopics\assetPricing2\data\FI\factor_database'
dirFig_database=r'D:\zht\database\quantDb\researchTopics\assetPricing2\data\FI\fig_database'
dirChen=r'D:\zht\database\quantDb\researchTopics\assetPricing2\data\FI\Chen2010'
dirPanels=r'D:\zht\database\quantDb\researchTopics\assetPricing2\data\FI\bivariate_panels'

def _save(df,name):
    df.to_pickle(os.path.join(dirData, name + '.pkl'))

def _read(name):
    return pd.read_pickle(os.path.join(dirData, name + '.pkl'))


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
            _save(df,'{}__{}'.format(tbname,varname))

            print(tbname,varname)


def indicator2factor(indicator):
    sampleControl=False
    q=5

    # data lagged
    df = _read(indicator)
    s = df.stack()
    s.name = indicator
    weight = Database(sample_control=sampleControl).by_indicators(['weight'])
    datalagged = pd.concat([s, weight], axis=1)
    datalagged = datalagged.groupby('sid').shift(1)

    # data t
    datat = Database(sample_control=sampleControl).by_indicators(['stockEretM'])
    comb = pd.concat([datalagged, datat], axis=1)
    comb = comb.dropna()

    comb['g'] = comb.groupby('t', group_keys=False).apply(
        lambda df: assign_port_id(df[indicator], q))

    panel = comb.groupby(['t', 'g']).apply(
        lambda df: my_average(df, 'stockEretM', wname='weight')) \
        .unstack(level=['g'])

    factor = panel[q] - panel[1]
    factor.name=indicator
    factor.to_pickle(os.path.join(factorPath, '{}.pkl'.format(indicator)))

def _task(indicator):
    try:
        indicator2factor(indicator)
    except:
        with open(os.path.join(direc, 'failed.txt'), 'a') as f:
            f.write(indicator+'\n')
    print(indicator)

def multi_indicator2factor():
    indicators=[ind[:-4] for ind in os.listdir(dirData)]
    p=Pool(6)
    p.map(_task,indicators)


def analyse_corr():
    '''
    analyse the correlation between different factors constructed by soritng
    financial indicators.Since there may be some indicators share the same value.

    Returns:

    '''
    fns=os.listdir(factorPath)

    ss=[pd.read_pickle(os.path.join(factorPath, fn)) for fn in fns]
    comb=pd.concat(ss,axis=1)
    corr=comb.corr()

    cc=corr.corr()

    tri=corr.mask(np.triu(np.ones(corr.shape),k=0).astype(bool))

    tri=tri.stack().sort_values(ascending=False)

    tri=tri.reset_index()

    thresh=0.9

def plot_all(factorPath,figPath):
    fns = os.listdir(factorPath)

    ss = [pd.read_pickle(os.path.join(factorPath, fn)) for fn in fns]
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
        fig.savefig(os.path.join(figPath, indicator + '.png'))
        print(i)

    sp=pd.DataFrame(tup,columns=['indicator','sharpe'])
    return sp

def select_a_model():
    sharpe=pd.read_pickle(os.path.join(direc, 'sharpe.pkl'))
    indicator=sharpe['indicator'][0]
    factor=pd.read_pickle(os.path.join(factorPath, indicator + '.pkl'))

    ff3=read_unfiltered('ff3M')
    model=pd.concat([ff3[['rp','smb']],factor],axis=1)
    model=model.dropna()
    return model

def compare_model_with_ff3():
    ff3=BENCH.by_benchmark('ff3M')
    model=select_a_model()
    tableas=[]
    tablets=[]
    for bench in [ff3,model]:
        tablea,tablet=regression_details_5x5(bench)
        tableas.append(tablea)
        tablets.append(tablet)

    comba=pd.concat(tableas,axis=0,keys=['ff3','myModel'])
    combt=pd.concat(tablets,axis=0,keys=['ff3','myModel'])

    comba.to_csv(os.path.join(direc,'comba.csv'))
    combt.to_csv(os.path.join(direc,'combt.csv'))

# find anomalies

def get_all_factor_from_database():
    info=Database().info
    indicators=[ele for l in info.values() for ele in l]

    for indicator in indicators:
        factor=single_sorting_factor(indicator,q=10)
        factor.name=indicator
        factor.to_pickle(os.path.join(dirFactor_database,indicator+'.pkl'))
        print(indicator)

# get_all_factor_from_database()
indicators=['idio__idioVol_ff3_1M__D',
         'liquidity__amihud',
         'liquidity__turnover1',
         'momentum__R3M',
         'reversal__reversal',
         'skewness__skew_24M__D',
         'idio__idioVol_capm_1M__D',
         'inv__inv',
         'value__bm',
         'beta__D_1M',
         'op__op',
         'roe__roe'
            ]

# table III of Chen, L., and Zhang, L. (2010). A better three-factor model that explains more anomalies. Journal of Finance 65, 563–595.


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

def compare_different_models():
    # hxz is best
    ts=[]
    for factor in indicators:
        s=pd.read_pickle(os.path.join(dirFactor_database,factor+'.pkl'))
        mymodel=select_a_model()
        bs=list(Benchmark().info.keys())
        names=['pure','my']+bs
        benchs=[None,mymodel]+[Benchmark().by_benchmark(r) for r in bs]

        t=pd.Series([_riskAdjust(s,bench) for bench in benchs],index=names)
        ts.append(t)
        print(factor)

    df=pd.concat(ts, axis=1, keys=indicators)
    df.to_csv(os.path.join(dirChen,'intercept_tvalue.csv'))

    dic={}
    for col,s in df.items():
        m=s.abs().idxmin()
        if m in dic:
            dic[m]+=1
        else:
            dic[m]=1


def get_bivariate_panel(v1, v2='size__size'):
    sampleControl = False
    q = 5

    ss=[]
    for v in [v1,v2]:
        if v in Database(sample_control=sampleControl).all_indicators:
            s=Database(sample_control=sampleControl).by_indicators([v])
        else:
            s=_read(v).stack()
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

    panel = comb.groupby(['t', 'g1', 'g2']).apply(
        lambda df: my_average(df, 'stockEretM', wname='weight'))\
        .unstack(level=['g1','g2'])
    print(v1)
    return panel

def _get_panel(indicator):
    panel=get_bivariate_panel(indicator)
    panel.to_pickle(os.path.join(dirPanels,'{}.pkl'.format(indicator)))

def multi_get_panel():
    multi_processing(_get_panel,indicators,pool_size=5)


def compare_models_based_on_bivariate_assets():
    resultLst=[]
    psLst=[]
    for indicator in indicators:
        assets=pd.read_pickle(os.path.join(dirPanels,'{}.pkl'.format(indicator)))
        mymodel = select_a_model()
        bs = list(Benchmark().info.keys())
        benchNames = ['pure', 'my'] + bs
        benchs = [None, mymodel] + [Benchmark().by_benchmark(r) for r in bs]

        rs=[]
        ps=[]
        for bench in benchs:
            r=ts_panel(assets,bench)
            rs.append(r)
            if bench is not None:
                p=model_performance(assets.copy(),bench)
                ps.append(p)
        resultLst.append(pd.concat(rs,axis=0,keys=benchNames))
        psLst.append(pd.concat(ps,axis=1,keys=benchNames[1:]).T)
        print(indicator)

    pd.concat(resultLst,axis=0,keys=indicators).to_csv(r'e:\a\result.csv')
    pd.concat(psLst,axis=0,keys=indicators).to_csv(r'e:\a\performance.csv')







# if __name__ == '__main__':
#     multi_get_panel()








