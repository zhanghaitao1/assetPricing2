# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-13  20:20
# NAME:assetPricing2-9market_status.py

'''
1. Li and Galvani, “Market States, Sentiment, and Momentum in the Corporate Bond Market.”
    page 254
2. Cooper Michael J., Gutierrez Roberto C., and Hameed Allaudeen, “Market States and Momentum.”
'''

from dout import read_df
import pandas as pd
from config import *
import os

def get_upDown():
    '''
    2. Cooper Michael J., Gutierrez Roberto C., and Hameed Allaudeen, “Market States and Momentum.”

    :return:
    '''
    mktRetM=read_df('mktRetM','M')
    windows=[12,24,36]
    series=[]
    for window in windows:
        s=mktRetM['mktRetM'].rolling(window=window).sum()
        s=s.shift(1)
        s[s>0]=1
        s[s<0]=-1
        series.append(s)

    upDown=pd.concat(series,axis=1,keys=windows)
    upDown.to_csv(os.path.join(DATA_PATH,'upDown.csv'))

def cal_market_states():
    '''
    market states:
        search for 'market state' in zoter
    1. Cheema and Nartea, “Momentum Returns, Market States, and Market Dynamics.”  chapter 3.1:
    Following Chui et al. (2010), we set stocks with monthly returns greater (lower) than 100 (−95) percent equal to 100
(−95) percent to avoid the influence of extreme returns and any possible data recording errors.

    :return:
    '''
    upDown=read_df('upDown','M')
    pass