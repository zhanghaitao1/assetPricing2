# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-18  22:54
# NAME:assetPricing2-observe_data.py

from tool import detect_outliers
from config import DATA_PATH

import pandas as pd
import os

def detect_outliers1(s,thresh=2):
    s=s.dropna()
    n=s.shape[0]
    avg=s.mean()
    std=s.std()
    up=(s>avg+thresh*std).sum()*100/n
    down=(s<avg-thresh*std).sum()*100/n
    return n,avg,s.max(),s.min(),up,down

df=pd.read_csv(os.path.join(DATA_PATH,'betaD.csv'),index_col=[0,1],parse_dates=True)
df.index.names=['type','t']
subdf=df.groupby('type').get_group('D_1M')

subdf=subdf.stack()
subdf=subdf.reset_index([0,2],drop=True)
subdf=subdf.dropna()

detect_outliers(subdf)

# for thresh in range(2,10):
#     print(thresh,detect_outliers1(subdf,thresh))


#TODO: stock sids
#TODO: detect outliers in src
#TODO: detect outliers in indicators
#TODO: detect outliers in factors#


