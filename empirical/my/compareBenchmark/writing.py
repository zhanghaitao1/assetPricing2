# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-21  20:37
# NAME:assetPricing2-writing.py
from data.dataApi import Database
from empirical.my.compareBenchmark.compare import database_indicators
from tool import summary_statistics
import pandas as pd

def get_descriptive_statistics():
    '''
    as table 2.3 (Bali et al., 2016)
    Returns:

    '''
    ss=[]
    for indicator in database_indicators:
        data=Database(sample_control=True).by_indicators([indicator])
        s=summary_statistics(data.unstack())
        ss.append(s.mean())

    table=pd.concat(ss,keys=database_indicators,axis=1)






