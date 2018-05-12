# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-05  23:45
# NAME:assetPricing2-ff5.py

'''
Preface:
    This module can be used to construct playing field as chapter 3 in "A
    five-factor asset pricing model" (Fama and French 2015)

references:
    1. Fama, E.F., and French, K.R. (1993). Common risk factors in the returns
    on stocks and bonds. Journal of Financial Economics 33, 3–56.
    2. Fama, E.F., and French, K.R. (2015). A five-factor asset pricing model.
    Journal of Financial Economics 116, 1–22.

'''
# playing field
import string
from itertools import combinations

from core.main import combine_with_datalagged
from core.constructFactor import data_for_bivariate
from data.base import MyError
from data.dataApi import Benchmark
from data.dataTools import load_data
from tool import my_average, GRS_test, assign_port_id
import pandas as pd
import statsmodels.formula.api as sm
import numpy as np
import os
import matplotlib.pyplot as plt
BENCH=Benchmark()
directory=r'D:\zht\database\quantDb\researchTopics\assetPricing2\empirical\ff5'

def _save(df,name):
    df.to_csv(os.path.join(directory,name+'.csv'))

def construct_playingField(vars,model):
    '''
    :param vars: list
    :param model: belong to {'5x5','2x4x4'}
    :return:
    '''
    if model=='5x5':
        v1,v2=tuple(vars)
        comb=data_for_bivariate(v1,v2,5,5,independent=True)
        assets=comb.groupby(['t','g1','g2']).apply(
            lambda df:my_average(df,'stockEretM',wname='weight'))\
            .unstack(level=['g1','g2'])
    elif model=='2x4x4':
        #v1 must belong to size category
        v1,v2,v3=tuple(vars)
        comb=combine_with_datalagged([v1,v2,v3])
        comb=comb.dropna()
        comb['g1'] = comb.groupby('t', group_keys=False).apply(
            lambda df: assign_port_id(df[v1], 2,range(1,3)))

        comb['g2'] = comb.groupby(['t','g1'], group_keys=False).apply(
            lambda df: assign_port_id(df[v2], 4,range(1,5)))

        comb['g3'] = comb.groupby(['t','g1'], group_keys=False).apply(
            lambda df: assign_port_id(df[v3], 4,range(1,5)))

        assets=comb.groupby(['t','g1','g2','g3']).apply(
            lambda df: my_average(df, 'stockEretM', wname='weight')) \
            .unstack(level=['g1', 'g2','g3'])
    else:
        raise MyError('Model "{}" is not supported currently'.format(model))

    return assets

def merge_panels(panels, titles):
    '''
    stack panels into one table,as table 1 of the paper.
    Args:
        panels:
        titles:

    Returns:

    '''
    newPanels=[]
    for i,(df,title) in enumerate(zip(panels, titles)):
        newIndex=['Panel {}: {}'.format(string.ascii_uppercase[i],title)]\
                 +df.index.tolist()
        newPanels.append(df.reindex(newIndex))
    table=pd.concat(newPanels,axis=0)
    return table

def get_table1():
    '''
    get table 1 in page 3
    Returns:

    '''
    v1='size__size'
    v2s=['value__bm','op__op','inv__inv']
    panels=[]
    titles=[]
    for v2 in v2s:
        assets=construct_playingField([v1,v2],'5x5')
        panel=assets.mean().unstack(level='g2')
        panel.index=['Small',2,3,4,'Big']
        panel.columns=['Low',2,3,4,'High']
        panels.append(panel)
        titles.append('{}-{}'.format(v1.split('__')[-1],v2.split('__')[-1]))
        print(v2)

    table=merge_panels(panels,titles)
    _save(table,'table1')

def get_table2():
    v1='size__size'
    v23s=[('value__bm','op__op'),('value__bm','inv__inv'),
          ('op__op','inv__inv')]

    smallPanels=[]
    bigPanels=[]
    titles=[]
    for v2,v3 in v23s:
        assets=construct_playingField([v1,v2,v3],'2x4x4')

        panel=assets.mean()
        panel.unstack(level='g1').head()
        small=panel[1].unstack(level='g3')
        big=panel[2].unstack(level='g3')
        small.index=['Low {}'.format(v2.split('__')[-1]),2,3,
                     'High {}'.format(v2.split('__')[-1])]
        small.columns=['Low',2,3,'High']

        big.index=['Low {}'.format(v2.split('__')[-1]),2,3,
                     'High {}'.format(v2.split('__')[-1])]
        big.columns=['Low',2,3,'High']

        smallPanels.append(small)
        bigPanels.append(big)
        titles.append('portfolios formed on Size,{},and {}'.format(
            v2.split('__')[-1],v3.split('__')[-1]
        ))
        print(v2,v3)

    smallTable=merge_panels(smallPanels,titles)
    bigTable=merge_panels(bigPanels,titles)
    _save(smallTable,'table2_small')
    _save(bigTable,'table2_big')

def model_performance(panel,bench):
    '''
    calculate the indicators such as GRS,and so on to compare the different
    models based on some assets. For details about these indicators refer to
    table 5 of "A five-factor asset pricing model" (Fama and French 2015).

    For the returned five indicators,except for the "grsp",for the other
    four indicators,the smaller the values,the better the model can
    explain the cross sectional returns of the assets
    Args:
        assets:assets constructed by function construct_playingField
        riskmodel: refer to BENCH.by_benchmark

    Returns:

    '''
    # make sure that we do not change the panel,since panel may be reused
    # outside this function
    assets=panel.copy()
    # change the column names,since sm.ols do not support numerical regressor.
    anames=['a{}'.format(i) for i in range(1,assets.shape[1]+1)]
    assets.columns=anames
    comb=pd.concat([assets,bench],axis=1)
    comb=comb.dropna()

    #GRS
    resid=[]
    alpha=[]
    for a in anames:
        formula='{} ~ {}'.format(a,' + '.join(bench.columns))
        r=sm.ols(formula,comb).fit()
        resid.append(r.resid)
        alpha.append(r.params['Intercept'])

    factor=comb[bench.columns].values
    resid=pd.concat(resid,axis=1).values
    alpha=np.asarray(alpha).reshape((resid.shape[1],1))
    grs,grsp=GRS_test(factor,resid,alpha)

    #A|a_i|
    Aai=abs(alpha).mean()

    #(A|a_i|)/(A|r_i|)
    Ri=assets.mean()
    R=Ri.mean()
    ri=Ri-R
    ratio1=Aai/abs(ri).mean()

    # ratio2
    ratio2=np.mean(np.square(alpha))/np.mean(np.square(ri))

    return pd.Series([grs,grsp,Aai,ratio1,ratio2],
                     index=['grs','grsp','Aai','ratio1','ratio2'])

def get_table5():
    # for 5x5 panels
    v1='size__size'
    v2s = ['value__bm', 'op__op', 'inv__inv']
    panels1=[]
    titles1=[]
    for v2 in v2s:
        assets=construct_playingField([v1,v2],'5x5')
        riskmodels=BENCH.info.keys()
        rows=[]
        for rm in riskmodels:
            rows.append(model_performance(assets,BENCH.by_benchmark(rm)))
        panel=pd.concat(rows,axis=1,keys=riskmodels).T
        panels1.append(panel)
        titles1.append('25 {}-{} portfolios'.format(v1.split('__')[-1],
                                                   v2.split('__')[-1]))
        print(v2)

    # For 2x4x4 panels
    v1='size__size'
    v23s=[('value__bm','op__op'),('value__bm','inv__inv'),
          ('op__op','inv__inv')]

    panels2=[]
    titles2=[]
    for v2, v3 in v23s:
        assets = construct_playingField([v1, v2, v3], '2x4x4')
        riskmodels=BENCH.info.keys()
        rows=[]
        for rm in riskmodels:
            rows.append(model_performance(assets,rm))
        panel=pd.concat(rows,axis=1,keys=riskmodels).T
        panels2.append(panel)
        titles2.append('32 {}-{}-{} portfolios'.format(
            v1.split('__')[-1],v2.split('__')[-1],v3.split('__')[-1]))

        print(v2,v3)

    panels=panels1+panels2
    titles=titles1+titles2

    table=merge_panels(panels,titles)
    _save(table,'table5')

def detect_redundant_factor(riskmodel='ff5'):
    bench=BENCH.by_benchmark(riskmodel)
    panels=[]
    for left in bench.columns:
        right=[col for col in bench.columns if col!=left]
        formula='{} ~ {}'.format(left,' + '.join(right))
        reg=sm.ols(formula,bench).fit()
        panel=pd.DataFrame([reg.params,reg.tvalues],index=['coef','t'])
        panel['r2']=reg.rsquared_adj
        panels.append(panel)

    table=pd.concat(panels,axis=0)
    return table

def get_table6():
    riskmodels = ['ff3','ffc','ff5','hxz4']
    tables=[detect_redundant_factor(rm) for rm in riskmodels]
    _save(pd.concat(tables,axis=0,keys=riskmodels),'table6')

def orthogonalize(riskmodel,factor):
    bench=BENCH.by_benchmark(riskmodel)
    others=[col for col in bench.columns.tolist() if col != factor]
    formula='{} ~ {}'.format(factor,' + '.join(others))
    r=sm.ols(formula,bench).fit()
    orth=r.resid+r.params['Intercept']
    return orth

def test_orthogonalize():
    factor='ff5M__hml'
    riskmodel='ff5'
    orth=orthogonalize(riskmodel,factor)

def _details_for_intercept(s, bench=None):
    s.name= 'y'
    s=s.to_frame()
    if bench is not None:
        comb = pd.concat([s, bench], axis=1)
        comb = comb.dropna()
        formula='y ~ {}'.format(' + '.join(bench.columns))
        reg=sm.ols(formula,comb).fit()
    else:
        formula='y ~ 1'
        reg=sm.ols(formula, s).fit()
    return pd.Series([reg.params['Intercept'],reg.tvalues['Intercept']],
                     index=['alpha','t'])


def ts_panel(panel,bench):
    return panel.apply(lambda s:_details_for_intercept(s,bench))

def regression_details_5x5(bench):
    '''
    as table 7 in page 13
    Args:
        bench:

    Returns:

    '''
    v1='size__size'
    v2s = ['value__bm', 'op__op', 'inv__inv']

    panelas1=[]
    panelts1=[]
    titles1=[]
    for v2 in v2s:
        assets=construct_playingField([v1,v2],'5x5')
        panela=pd.DataFrame()
        panelt=pd.DataFrame()
        for col,s in assets.items():
            result=_details_for_intercept(s,bench)
            panela.at[col[0],col[1]]=result['alpha']
            panelt.at[col[0],col[1]]=result['t']
        panela.index=['Small',2,3,4,'Big']
        panela.columns=['Low',2,3,4,'High']

        panelt.index=['Small',2,3,4,'Big']
        panelt.columns=['Low',2,3,4,'High']

        panelas1.append(panela)
        panelts1.append(panelt)
        titles1.append('25 {}-{} portfolios'.format(v1.split('__')[-1],
                                                    v2.split('__')[-1]))

    tablea=merge_panels(panelas1,titles1)
    tablet=merge_panels(panelts1,titles1)
    return tablea,tablet

def regression_details_2x4x4(bench):
    '''
    as table 11 in page 18
    Args:
        bench:

    Returns:

    '''
    v1 = 'size__size'
    v23s = [('value__bm', 'op__op'), ('value__bm', 'inv__inv'),
            ('op__op', 'inv__inv')]

    panelas_small = []
    panelts_small = []
    panelas_big = []
    panelts_big = []

    titles2 = []
    for v2, v3 in v23s:
        assets = construct_playingField([v1, v2, v3], '2x4x4')
        panela_small = pd.DataFrame()
        panelt_small = pd.DataFrame()

        panela_big = pd.DataFrame()
        panelt_big = pd.DataFrame()
        for col, s in assets.items():
            result=_details_for_intercept(s,bench)
            if col[0] == 1:
                panela_small.at[col[1], col[2]] = result['alpha']
                panelt_small.at[col[1], col[2]] = result['t']
            elif col[0] == 2:
                panela_big.at[col[1], col[2]] = result['alpha']
                panelt_big.at[col[1], col[2]] = result['t']

        panela_small.index = ['Low {}'.format(v2.split('__')[-1]), 2, 3,
                              'High {}'.format(v2.split('__')[-1])]
        panela_small.columns = ['Low', 2, 3, 'High']
        panelt_small.index = ['Low {}'.format(v2.split('__')[-1]), 2, 3,
                              'High {}'.format(v2.split('__')[-1])]
        panelt_small.columns = ['Low', 2, 3, 'High']

        panela_big.index = ['Low {}'.format(v2.split('__')[-1]), 2, 3,
                            'High {}'.format(v2.split('__')[-1])]
        panela_big.columns = ['Low', 2, 3, 'High']
        panelt_big.index = ['Low {}'.format(v2.split('__')[-1]), 2, 3,
                            'High {}'.format(v2.split('__')[-1])]
        panelt_big.columns = ['Low', 2, 3, 'High']

        panelas_small.append(panela_small)
        panelts_small.append(panelt_small)

        panelas_big.append(panela_big)
        panelts_big.append(panelt_big)

        titles2.append('32 {}-{}-{} portfolios'.format(
            v1.split('__')[-1], v2.split('__')[-1], v3.split('__')[-1]))

    tablea_small = merge_panels(panelas_small, titles2)
    tablet_small = merge_panels(panelts_small, titles2)

    tablea_big = merge_panels(panelas_big, titles2)
    tablet_big = merge_panels(panelts_big, titles2)
    return tablea_small,tablet_small,tablea_big,tablet_big

def get_table7():
    riskmodels=list(BENCH.info.keys())
    tableas_=[]
    tablets_=[]
    for rm in riskmodels:
        bench=BENCH.by_benchmark(rm)
        tablea,tablet=regression_details_5x5(bench)
        tableas_.append(tablea)
        tablets_.append(tablet)
        print(rm)
    comba=pd.concat(tableas_,axis=0,keys=riskmodels)
    combt=pd.concat(tablets_,axis=0,keys=riskmodels)
    _save(comba,'table7_5x5_alpha')
    _save(combt,'table7_5x5_talpha')

    tables_ll=[]
    for rm in riskmodels:
        # tablea_small,tablet_small,tablea_big,tablet_big=regression_details_2x4x4(rm)
        bench=BENCH.by_benchmark(rm)
        tables_ll.append(regression_details_2x4x4(bench))
        print(rm)

    _save(pd.concat([t[0] for t in tables_ll],axis=0,keys=riskmodels),
          'table7_2x4x4_small_alpha')
    _save(pd.concat([t[1] for t in tables_ll],axis=0,keys=riskmodels),
          'table7_2x4x4_small_talpha')
    _save(pd.concat([t[2] for t in tables_ll],axis=0,keys=riskmodels),
          'table7_2x4x4_big_alpha')
    _save(pd.concat([t[3] for t in tables_ll],axis=0,keys=riskmodels),
          'table7_2x4x4_big_talpha')

if __name__ == '__main__':
    get_table1()
    get_table2()
    get_table5()
    get_table6()
    get_table7()



'''
Ideas:
1. split samples into sz and sh
2. split into two subsample
3. add FF6 and so on
4. industry indices
5. explain funds
6. refine hxz4 and capm

Concerns:
1. this seems to be playing home game,

Results:
1. table5: capm and hxz4 is really abnormal
2. table6: hxz4 the right hand variables may be correlated with each other
3. table7:capm and hxz is bad

The results seem to be that capm and hxz4 are highly similar,and the other three models,ff3,ffc,ff5 are almost equivalent.	

'''
