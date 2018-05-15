# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-14  10:27
# NAME:assetPricing2-analyse_result.py

from multiprocessing.pool import Pool

from core.constructFactor import get_single_sorting_assets
from core.ff5 import ts_panel, model_performance
from data.dataApi import Database, Benchmark
from data.dataTools import read_unfiltered
from data.din import parse_financial_report, toMonthly
from empirical.my.compareBenchmark.playingField import _get_reduced_indicators, \
    get_significant_indicators
from tool import assign_port_id, my_average, newey_west, multi_processing, \
    get_riskAdjusted_alpha_tvalue
from zht.data.gta.api import read_gta
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zht.utils.listu import group_with

dirProj= r'D:\zht\database\quantDb\researchTopics\assetPricing2\my'
dirFI=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\financial_indicators'
dir10assets=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\10assets'
dirSpread= r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\10minus1'
dirSpreadFig=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\spread_fig'
dirDatabaseAssets=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\databaseAssets'
dirDatabaseSpread=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\database_10Minus1'
dirCompare=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\compare_models'
dir25assets=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\25assets'
dirDatabaseSpreadFig=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\databaseSpread_fig'
dirIndustryIndex=r'D:\zht\database\quantDb\researchTopics\assetPricing2\my\industryIndex'


def compare_significant_numbers():
    '''
    The result is rediculous
    Returns:

    '''
    thresh=2#TODO:
    tvalues=pd.read_csv(os.path.join(dirCompare,'spreadInterceptTvalues.csv'),
                        index_col=0)
    tvalues=tvalues.T
    for col,s in tvalues.items():
        tvalues[col+'_bool']=abs(s)>thresh
        tvalues[col+'_spread_abs']=abs(s)-abs(tvalues['ff3M'])
    tvalues.sum()

    # t test on the reduction of tvalues as panel C of table 2 in
    # (Huynh, 2017)
    a=tvalues.mean()/tvalues.std()



