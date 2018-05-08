# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-08  10:00
# NAME:assetPricing2-11 FF6.py
from core.constructFactor import data_for_bivariate, two_sorting_factor
from core.main import combine_with_datalagged
from data.dataTools import read_unfiltered, save
import matplotlib.pyplot as plt
import pandas as pd

def get_ff6():
    v1='size__size'
    v2='momentum__r12'

    smb,mom=two_sorting_factor(v1,v2,2,[0,0.3,0.7,1.0],sample_control=False,
                               independent=True)
    mom.index.name='t'
    mom.name='mom'

    ff5=read_unfiltered('ff5M')
    ff6=pd.concat([ff5,mom],axis=1)
    ff6=ff6.dropna()
    ff6.columns.name='type'

    save(ff6,'ff6M')







