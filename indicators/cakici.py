# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-24  09:47
# NAME:assetPricing2-cakici.py
from data.dataTools import read_filtered, save
import pandas as pd
import numpy as np
from zht.data.gta.api import read_gta


def get_sd():
    ri=read_filtered('stockRetD')*100
    sd=ri.rolling(20,min_periods=15).std().resample('M').last()
    save(sd,'sd')

def get_cfpr():
    df=read_gta('STK_MKT_Dalyr',encoding='gbk')
    df['t']=pd.to_datetime(df['TradingDate'])
    df['sid']=df['Symbol'].astype(str)
    df['cfpr']=1.0/df['PCF']
    cfpr=pd.pivot_table(df,values='cfpr',index='t',columns='sid')
    cfpr=cfpr.sort_index().resample('M').last()
    save(cfpr,'cfpr')

def get_ep():
    df=read_gta('STK_MKT_Dalyr',encoding='gbk')
    df['t']=pd.to_datetime(df['TradingDate'])
    df['sid']=df['Symbol'].astype(str)
    df['ep']=1.0/df['PE']
    cfpr=pd.pivot_table(df,values='ep',index='t',columns='sid')
    cfpr=cfpr.sort_index().resample('M').last()
    save(cfpr,'ep')


