# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-24  17:34
# NAME:assetPricing2-indicators_filter.py

from data.dataTools import load_data
from data.outlier import delete_outliers, detect_outliers

def analyse_outliers(data):
    #TODO: a little messy
    detect_outliers(data, 'data')

    #---------------------------liquidity----------------------------------
    liquidity=load_data('liquidity')
    #amihud
    amihud=liquidity['amihud'].unstack()
    amihud=delete_outliers(amihud,method='percentile',thresh=98,pooled=True)
    detect_outliers(amihud, 'amihud')
    #filter ps1
    ps1=liquidity['ps1'].unstack()
    ps1_pooled=delete_outliers(ps1,method='percentile',thresh=99,pooled=True)
    detect_outliers(ps1_pooled, 'ps1_pooled')
    #ps2
    ps2=liquidity['ps2'].unstack()
    ps2_pooled=delete_outliers(ps2,method='percentile',thresh=99,pooled=True)
    detect_outliers(ps2_pooled, 'ps2_pooled')
    #roll1
    roll1=liquidity['roll1'].unstack()
    roll1_pooled=delete_outliers(roll1,method='percentile',thresh=99,pooled=True)
    detect_outliers(roll1_pooled, 'roll1_pooled')
    #roll2
    roll2=liquidity['roll2'].unstack()
    roll2_pooled=delete_outliers(roll2,method='percentile',thresh=99,pooled=True)
    detect_outliers(roll2_pooled, 'roll2_pooled')
    #zeros1
    zeros1=liquidity['zeros1'].unstack()
    zeros1_pooled=delete_outliers(zeros1,method='percentile',thresh=99,pooled=True)
    detect_outliers(zeros1_pooled, 'zeros1_pooled')
    #zeros2
    zeros2=liquidity['zeros2'].unstack()
    zeros2_pooled=delete_outliers(zeros2,method='percentile',thresh=99,pooled=True)
    detect_outliers(zeros2_pooled, 'zeros2_pooled')

    #----------------------------skewness----------------------------------------
    skewness=load_data('skewness')

    idioskew_24M__D=skewness['idioskew_24M__D'].unstack()
    idioskew_24M__D_pooled=delete_outliers(idioskew_24M__D,method='percentile',thresh=99.9,pooled=True)
    detect_outliers(idioskew_24M__D_pooled,'idioskew_24M__D_pooled')

    skew_12M__D=skewness['skew_12M__D'].unstack()
    skew_12M__D_pooled=delete_outliers(skew_12M__D,method='percentile',thresh=99.9,pooled=True)
    detect_outliers(skew_12M__D_pooled,'skew_12M__D_pooled')

    skew_24M__D=skewness['skew_24M__D'].unstack()
    skew_24M__D_pooled=delete_outliers(skew_24M__D,method='percentile',thresh=99.9,pooled=True)
    detect_outliers(skew_24M__D_pooled,'skew_24M__D_pooled')

def refine(data):

    dic={'liquidity':['amihud','ps1','ps2'],
         'skewness':['idioskew_24M__D','skew_12M__D','skew_24M__D']}
    for category,indicators in dic.items():
        for indicator in indicators:
            col='{}__{}'.format(category,indicator)
            initial = data[col]
            if category=='liquidity':
                new=delete_outliers(initial.unstack(),method='percentile',thresh=99,pooled=True).stack()
            elif category=='skewness':
                new = delete_outliers(initial.unstack(), method='percentile', thresh=99.9, pooled=True).stack()
            else:
                raise ValueError
            data[col]=new
            print('refine {} {}'.format(category,indicator))
    return data


