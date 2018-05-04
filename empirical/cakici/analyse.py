# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-08  17:49
# NAME:assetPricing2-analyse.py

import pandas as pd

from core.main import combine_with_datalagged, risk_adjust
from data.dataApi import Database
from data.sampleControl import cross_size_groups
from tool import assign_port_id, my_average

DATA=Database()
indicators=['size__size','beta__D_24M',
            'momentum__R6M','reversal__reversal',
            'liquidity__turnover1','skewness__skew_24M__D',
            'idio__vol_24M__D']


#table 1  panel A
# by_year=comb[indicators].resample('Y',level='t').agg(lambda x:x.mean())

#table 1 panel B
# corr=correlation_mixed(comb[indicators])


def get_a_panel(comb,sorting_var):
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

comb=combine_with_datalagged(indicators)

#table2
table2=pd.concat([get_a_panel(comb,var) for var in indicators],axis=0,keys=indicators)

#table 3
cs,cm,cb=cross_size_groups()
small=comb[cs]
table3=pd.concat([get_a_panel(small,var) for var in indicators],axis=0,keys=indicators)

table3a=table3['eqw'].unstack(level=0)
table3b=table3['vw'].unstack(level=0)

table2a=table2['eqw'].unstack(level=0)

table2a.head()
table3a.head()

small.notnull().sum().sum()
comb.notnull().sum().sum()


comb.shape

cs.shape

test=comb[cs]
test.shape

test1=comb[c]

cs.sum()

test=comb.index.intersection(cs.dropna().index)

test[:5]
