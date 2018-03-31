# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-21  09:12
# NAME:assetPricing2-dout.py

import os
import pandas as pd
from config import DATA_PATH
from pandas.tseries.offsets import MonthEnd


def read_df(fn, freq):
    '''
    read df from DATA_PATH

    :param fn:
    :return:
    '''
    if not fn.endswith('.csv'):
        fn=fn+'.csv'
    df=pd.read_csv(os.path.join(DATA_PATH, fn), index_col=0)
    if freq=='M':
        df.index=pd.to_datetime(df.index)+MonthEnd(0)
        # df.index=pd.to_datetime(df.index).to_period(freq).to_timestamp(freq)
    elif freq=='D':
        df.index=pd.to_datetime(df.index)

    df.index.name='t'
    return df


