# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-05  15:40
# NAME:assetPricing2-timeSeriesRegression.py


#construct new factors
from core.main import combine_with_datalagged
from data.base import MyError
from tool import assign_port_id, my_average
import matplotlib.pyplot as plt


def single_sorting_factor(indicator, q, weight=False):
    # method1 independent way
    if isinstance(q, int):
        labels = ['g{}'.format(i) for i in range(1, q + 1)]
    elif isinstance(q, (list,tuple)):
        labels = ['g{}'.format(i) for i in range(1, len(q))]
    else:
        raise MyError('q:"{}"  is wrong!'.format(repr(q)))

    spreadname='_'.join([labels[-1],labels[0]])
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

def data_for_bivariate(v1, v2, q1, q2, independent=True):
    comb=combine_with_datalagged([v1,v2])
    comb=comb.dropna()

    if independent:
        comb['g1'] = comb.groupby('t', group_keys=False).apply(
            lambda df: assign_port_id(df[v1], q1,range(1,q1+1)))

        comb['g2'] = comb.groupby('t', group_keys=False).apply(
            lambda df: assign_port_id(df[v2], q2,range(1,q2+1)))
    else: #dependent
        '''
        v2 is conditional on v1,that is,we first group stocks into n1 portfolios
        based on the rank of v1,and then we will group each of the n1 portfolios
        into n2 portfolios based on v2
        '''
        comb['g1'] = comb.groupby('t', group_keys=False).apply(
            lambda df: assign_port_id(df[v1], q1, range(1,q1+1)))

        comb['g2'] = comb.groupby(['t', 'g1'], group_keys=False).apply(
            lambda df: assign_port_id(df[v2], q2, range(1,q2+1)))

    return comb

def two_sorting_factor(v1, v2, q1, q2,independent=True, weight=True):
    comb=data_for_bivariate(v1, v2, q1, q2, independent=independent)

    if weight:
        s=comb.groupby(['t','g1','g2']).apply(
            lambda df: my_average(df, 'stockEretM', wname='weight'))
    else:
        s = comb.groupby(['t', 'g1','g2'])['stockEretM'].mean()

    panel1=s.groupby(['t','g1']).mean().unstack(level='g1')
    factor1=panel1[q1]-panel1[1]

    panel2=s.groupby(['t','g2']).mean().unstack(level='g2')
    factor2=panel2[q2]-panel2[1]
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

def test_single_way():
    indicator='size__size'
    factor=single_sorting_factor(indicator,q=(0,0.3,0.9,1),weight=True)

def test_two_independent_way():
    q1 = 3
    q2 = 3
    v1 = 'skewness__idioskew_24M__D'
    v2 = 'size__size'

    for indepedent in [True,False]:
        for weight in[True,False]:
            factor1,factor2=two_sorting_factor(v1, v2, q1, q2,indepedent, weight)

            fig=plt.figure()
            plt.plot(factor1.index,factor1.cumsum())
            fig.savefig(r'e:\a\factor1_{}_{}'.format(repr(indepedent),repr(weight)))

            fig=plt.figure()
            plt.plot(factor2.index,factor2.cumsum())
            fig.savefig(r'e:\a\factor2_{}_{}'.format(repr(indepedent),repr(weight)))

def test_three_sorting():
    v1 = 'skewness__idioskew_24M__D'
    v2 = 'size__size'  # controlling factor1
    v3 = 'value__logbm'  # controlling factor2
    q1 = q2 = q3 = 3

    factor1,factor2,factor3=three_sorting_factor(v1, v2, v3, q1, q2, q3)


def get_SMB_HML():
    v1='size__size'
    v2='value__logbm'
    q1=2
    q2=3
    smb,hml=two_sorting_factor(v1,v2,q1,q2,independent=True,weight=True)
    fig=plt.figure()
    plt.plot(smb.index,(smb*(-1)).cumsum())
    plt.show()

    fig=plt.figure()
    plt.plot(hml.index,hml.cumsum())
    plt.show()

# get_SMB_HML()


#TODO: Here,we update the portfolios monthly,but the indicators related to
# financial report are updately quartly or yearly,So,our portoflios contructed
# by ranking these indicators should update quartly or yearly rather than monthly
# How to handle this problem.

'''
maybe,we do not need to care about this problem,if we ffill the indicators.

'''















