# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-06  22:39
# NAME:assetPricing2-test_constructFactor.py
from core.constructFactor import single_sorting_factor, two_sorting_factor, \
    three_sorting_factor
import matplotlib.pyplot as plt

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


