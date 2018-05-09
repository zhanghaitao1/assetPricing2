# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-08  17:49
# NAME:assetPricing2-artmann.py

import pandas as pd
import os
from itertools import combinations

from core.main import combine_with_datalagged, risk_adjust, Bivariate
from data.dataApi import Database
from data.sampleControl import cross_size_groups, control_sid
from empirical.cakici.get_data import PATH
from tool import assign_port_id, my_average, correlation_mixed

DATA=Database()
indicators=['size__size','beta__D_24M','value__logbm',
            'momentum__R6M','reversal__reversal',
            'liquidity__turnover1','skewness__skew_24M__D',
            'idio__vol_24M__D']

all_=combine_with_datalagged(indicators)

def get_table1():
    #table 1  panel A
    by_year=all_[indicators].resample('Y',level='t').agg(lambda x:x.mean())

    #table 1 panel B
    corr=correlation_mixed(all_[indicators])
    by_year.to_csv(os.path.join(PATH,'table1_panelA'))
    corr.to_csv(os.path.join(PATH,'table1_panelB'))


def _get_a_panel(comb, sorting_var):
    q=5
    gcol = 'g_%s' % sorting_var
    gnames=['g{}'.format(i) for i in range(1,q+1)]
    spreadname='_'.join([gnames[-1],gnames[0]])
    comb[gcol] = comb.groupby('t', group_keys=False).apply(
        lambda df: assign_port_id(df[sorting_var], q,gnames))

    # ind=[ind for ind in indicators if ind not in
    #         [sorting_var,'stockEretM','weight']]

    ss=[]
    for ind in indicators:
        s=comb.groupby(['t',gcol])[ind].mean().groupby(gcol).mean()
        ss.append(s)

    partB=pd.concat(ss,axis=1,keys=indicators)

    eqw=comb.groupby(['t', gcol])['stockEretM'].mean().unstack()
    vw=comb.groupby(['t',gcol]).apply(
        lambda df:my_average(df,'stockEretM',wname='weight')).unstack()

    as_=[]
    for w,name in zip([eqw,vw],['eqw','vw']):
        a1=w.mean()
        a1.index=a1.index.astype(str)
        w.columns=w.columns.astype(str)
        w[spreadname]=w[gnames[-1]]-w[gnames[0]]
        a2=risk_adjust(w[spreadname],riskmodels=[None,'capm','ff3'])
        a=a1.append(a2)
        a.name=name
        as_.append(a)

    result=pd.concat(as_+[partB],axis=1)
    result=result.reindex(index=as_[0].index)
    print(sorting_var)
    return result

def get_table2():
    #table2
    table2=pd.concat([_get_a_panel(all_, var) for var in indicators],
                     axis=0, keys=indicators)
    table2.to_csv(os.path.join(PATH,'table2.csv'))

def get_table3():
    #table 3
    s,m,b=cross_size_groups()

    tables=[]
    for marker in [s,m,b]:
        cohort=all_[marker.stack()]
        t=pd.concat([_get_a_panel(cohort, var) for var in indicators],
                    axis=0, keys=indicators)
        teqw=t['eqw'].unstack(level=0)
        tvw=t['vw'].unstack(level=0)
        tables.append(pd.concat([teqw,tvw],axis=0,keys=['eqw','vw']))

    table3=pd.concat(tables,axis=0,keys=['small','medium','big'])
    table3.to_csv(os.path.join(PATH,'table3.csv'))

def get_table4():
    indeVars_lst=sum([list(combinations(indicators,i)) for i in range(1,len(indicators)+1)],[])
    #TODO:what's the standard for choosing multiple regressors?

    params_lst=[]
    for indeVars in indeVars_lst:
        params,_=Bivariate.famaMacbeth_reg(list(indeVars))
        params_lst.append(params)
        print(indeVars)

    index_order=params_lst[-1].index
    table4=pd.concat(params_lst,axis=1)
    table4=table4.reindex(index=index_order)
    table4.to_csv(os.path.join(PATH,'table4.csv'))

def get_table5():
    # partition into SHSE and SZSE
    idx=pd.IndexSlice
    tables=[]
    for cohort in ['is_sh','is_sz']:
        sids=control_sid(cohort)
        sub=all_.loc[idx[:,sids],]
        t = pd.concat([_get_a_panel(sub, var) for var in indicators],
                      axis=0, keys=indicators)
        teqw = t['eqw'].unstack(level=0)
        tvw = t['vw'].unstack(level=0)
        tables.append(pd.concat([teqw, tvw], axis=0, keys=['eqw', 'vw']))
    table5=pd.concat(tables,axis=0,keys=['sh','sz'])
    table5.to_csv(os.path.join(PATH, 'table5.csv'))


def run():
    get_table1()
    get_table2()
    get_table3()
    get_table4()
    get_table5()


if __name__ == '__main__':
    run()
