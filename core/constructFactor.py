# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-05  15:40
# NAME:assetPricing2-constructFactor.py

'''
preface:
    This module is used to construct factors by sorting indicators and long the
    top portfolio and short the bottom portfolio.

    For example,function two_sorting_factor() can be used to calculate SMB and
    HML in FF3. Function three_sorting_factor() can be used to consturct ATR in
    "Speculative and Trading and stock returns" (Pan et al 2016).


Notice:
    1. Some portfolio update yearly (like book-to-market) and some update monthly,
    However,we do not need to care about this problem,since we have handle the
    time problem at the beginning of calculating indicators.That is,for those
    yearly updated indicators, we have solved this problem by filtering out those
    data not appeared in the year end and filled na forwardly.We have also take
    data-available problem into consideration by shifting data forward with 6 months.



'''

from core.main import combine_with_datalagged
from data.base import MyError
from tool import assign_port_id, my_average
import matplotlib.pyplot as plt

def get_single_sorting_assets(indicator, q, weight=True):
    if isinstance(q, int):
        labels = ['g{}'.format(i) for i in range(1, q + 1)]
    elif isinstance(q, (list,tuple)):
        labels = ['g{}'.format(i) for i in range(1, len(q))]
    else:
        raise MyError('q:"{}"  is wrong!'.format(repr(q)))

    comb=combine_with_datalagged([indicator])
    comb['g'] = comb.groupby('t', group_keys=False).apply(
        lambda df: assign_port_id(df[indicator], q, labels))

    if weight:
        assets=comb.groupby(['t','g']).apply(
            lambda df:my_average(df,'stockEretM',wname='weight'))\
            .unstack(level=['g'])
    else:
        assets=comb.groupby(['t','g'])['stockEretM'].mean().unstack(level=['g'])
    return assets

def single_sorting_factor(indicator, q, weight=False):
    # method1 independent way
    '''
    This function is used to construct a new factor by a given indicator.
    We first group stocks into "q" portfolios based on the rank of "indicator"
    every month.Then,at the next month we calculate the corresponding monthly
    value-weighted (if weight is True) portfolio return.The factor return is
    the spread between the return of the top portfolio and bottom portfolio.

    :param indicator:
    :param q:
    :param weight:
    :return:Series
    '''
    if isinstance(q, int):
        labels = ['g{}'.format(i) for i in range(1, q + 1)]
    elif isinstance(q, (list,tuple)):
        labels = ['g{}'.format(i) for i in range(1, len(q))]
    else:
        raise MyError('q:"{}"  is wrong!'.format(repr(q)))

    comb=combine_with_datalagged([indicator])
    comb['g'] = comb.groupby('t', group_keys=False).apply(
        lambda df: assign_port_id(df[indicator], q, labels))

    if weight:
        panel=comb.groupby(['t','g']).apply(
            lambda df:my_average(df,'stockEretM',wname='weight'))\
            .unstack(level=['g'])
    else:
        panel=comb.groupby(['t','g'])['stockEretM'].mean().unstack(level=['g'])

    factor=panel[labels[-1]]-panel[labels[0]]
    return factor

def data_for_bivariate(v1, v2, q1, q2,independent=True,**kwargs):
    comb=combine_with_datalagged([v1,v2],**kwargs)
    comb=comb.dropna()

    if independent:
        comb['g1'] = comb.groupby('t', group_keys=False).apply(
            lambda df: assign_port_id(df[v1], q1))

        comb['g2'] = comb.groupby('t', group_keys=False).apply(
            lambda df: assign_port_id(df[v2], q2))
    else: #dependent
        '''
        v2 is conditional on v1,that is,we first group stocks into n1 portfolios
        based on the rank of v1,and then we will group each of the n1 portfolios
        into n2 portfolios based on v2
        '''
        comb['g1'] = comb.groupby('t', group_keys=False).apply(
            lambda df: assign_port_id(df[v1], q1))

        comb['g2'] = comb.groupby(['t', 'g1'], group_keys=False).apply(
            lambda df: assign_port_id(df[v2], q2))

    return comb

def two_sorting_factor(v1, v2, q1, q2,independent=True, weight=True,**kwargs):
    '''
    just like the way we construct SMB and HML

    :param v1:
    :param v2:
    :param q1:
    :param q2:
    :param independent: sort independently or not
    :param weight:
    :return: a tuple of two Series
    '''
    comb=data_for_bivariate(v1, v2, q1, q2,independent=independent,**kwargs)

    if weight:
        s=comb.groupby(['t','g1','g2']).apply(
            lambda df: my_average(df, 'stockEretM', wname='weight'))
    else:
        s = comb.groupby(['t', 'g1','g2'])['stockEretM'].mean()

    panel1=s.groupby(['t','g1']).mean().unstack(level='g1')
    factor1=panel1[panel1.columns.max()]-panel1[1]

    panel2=s.groupby(['t','g2']).mean().unstack(level='g2')
    factor2=panel2[panel2.columns.max()]-panel2[1]
    return factor1,factor2

def three_sorting_factor(v1, v2, v3, q1, q2, q3, weight=True):
    '''

    v1 and v2 are independent,v3 is conditional on v1 and v2

    reference:
        page 18 of Pan, L., Tang, Y., and Xu, J. (2016).
        Speculative Trading and Stock Returns. Review of Finance 20, 1835–1865.


    '''

    comb = combine_with_datalagged([v1, v2,v3])
    comb = comb.dropna()

    comb['g1'] = comb.groupby('t', group_keys=False).apply(
        lambda df: assign_port_id(df[v1], q1, range(1, q1 + 1)))

    comb['g2'] = comb.groupby('t', group_keys=False).apply(
        lambda df: assign_port_id(df[v2], q2, range(1, q2 + 1)))

    # g3 is conditional on g1 and g2
    comb['g3']=comb.groupby(['t','g1','g2'],group_keys=False).apply(
        lambda df: assign_port_id(df[v3], q3, range(1,q3+1)))

    if weight:
        s=comb.groupby(['t','g1','g2','g3']).apply(
            lambda df: my_average(df, 'stockEretM', wname='weight'))
    else:
        s=comb.groupby(['t','g1','g2','g3'])['stockEretM'].mean()

    panel1=s.groupby(['t','g1']).mean().unstack(level='g1')
    factor1=panel1[q1]-panel1[1]

    panel2=s.groupby(['t','g2']).mean().unstack(level='g2')
    factor2=panel2[q2]-panel2[1]

    panel3=s.groupby(['t','g3']).mean().unstack(level='g3')
    factor3=panel3[q3]-panel3[1]
    return factor1,factor2,factor3
















