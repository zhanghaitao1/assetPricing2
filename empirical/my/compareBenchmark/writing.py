# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-21  20:37
# NAME:assetPricing2-writing.py


from data.dataApi import Database
from empirical.my.compareBenchmark.compare import DATABASE_INDICATORS
from tool import summary_statistics, correlation_mixed
import pandas as pd
import os

#writing path
WP=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\writing'

def _save(df,name):
    df.to_csv(os.path.join(WP,name+'.csv'))

def get_descriptive_statistics():
    '''
    as table 2.3 (Bali et al., 2016)

    The result seem to be noise, as shown in the column of kurt
    '''
    ss=[]
    for indicator in DATABASE_INDICATORS:
        data=Database().by_indicators([indicator])
        s=summary_statistics(data.unstack())
        ss.append(s.mean())

    table=pd.concat(ss, keys=DATABASE_INDICATORS, axis=1).T
    _save(table,'descriptive')

def get_correlation():
    '''get correlation as table 8.2 (Bali et al.,2016)'''
    data=Database().by_indicators(DATABASE_INDICATORS)
    corr=correlation_mixed(data)
    _save(corr,'factor_correlation')

#TODO: filter out abnormal prior to descriptive statistics

ind='roe__roe'
data=Database().by_indicators([ind])

import seaborn as sns
sns.distplot(data.values)
import matplotlib.pyplot as plt



