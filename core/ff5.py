# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-05  23:45
# NAME:assetPricing2-ff5.py

'''
references:
    1. Fama, E.F., and French, K.R. (1993). Common risk factors in the returns
    on stocks and bonds. Journal of Financial Economics 33, 3–56.
    2. Fama, E.F., and French, K.R. (2015). A five-factor asset pricing model.
    Journal of Financial Economics 116, 1–22.

'''




# playing field
from core.main import combine_with_datalagged
from core.timeSeriesRegression import data_for_bivariate
from data.base import MyError
from data.dataApi import Benchmark
from data.dataTools import load_data
from tool import my_average, GRS_test, assign_port_id
import pandas as pd
import statsmodels.formula.api as sm
import numpy as np

BENCH=Benchmark()

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

        comb['g2'] = comb.groupby('t', group_keys=False).apply(
            lambda df: assign_port_id(df[v2], 4,range(1,5)))

        comb['g3'] = comb.groupby('t', group_keys=False).apply(
            lambda df: assign_port_id(df[v3], 4,range(1,5)))

        assets=comb.groupby(['t','g1','g2','g3']).apply(
            lambda df: my_average(df, 'stockEretM', wname='weight')) \
            .unstack(level=['g1', 'g2','g3'])
    else:
        raise MyError('Model "{}" is not supported currently'.format(model))

    #asset names
    anames=['a{}'.format(i) for i in range(1,assets.shape[1]+1)]
    assets.columns=anames
    return assets

def get_table5(vars,model,riskmodel):
    '''
    table 5 of FF5, at page 9
    :param vars:
    :param model:refer to construct_playingField
    :param riskmodel:refer to BENCH.by_benchmark
    :return:
    '''
    assets=construct_playingField(vars,model)
    anames=assets.columns.tolist()# asset names
    bench=BENCH.by_benchmark(riskmodel)
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
        print(a)

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

    return grs,Aai,ratio1,ratio2

def test_table5():
    v1='size__size'
    v2='value__logbm'
    grs,Aai,ratio1,ratio2=get_table5([v1,v2],'5x5',riskmodel='ff3')

    print(grs,Aai,ratio1,ratio2)

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

