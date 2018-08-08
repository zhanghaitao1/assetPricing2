# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-21  10:02
# NAME:assetPricing2-1 beta_new.py
import multiprocessing

from data.dataTools import read_unfiltered,save
import pandas as pd
import numpy as np
from zht.utils.dfu import myroll, my_rolling_apply


def _cal_beta(df, min_periods):
    df=df.dropna(thresh=min_periods, axis=1)
    df=df.fillna(df.mean()) #Trick: fillna with average
    # df=df.fillna(0)
    # first column is the market
    X = df.values[:, [0]]
    # prepend a column of ones for the intercept
    X = np.concatenate([np.ones_like(X), X], axis=1)
    # matrix algebra
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, 1:])
    return pd.Series(b[1],index= df.columns[1:], name='beta')

def task(args):
    print(args[0],args[2])
    comb=args[1]
    window=args[2]
    min_periods=int(window*3/4)
    s=myroll(comb,window).apply(_cal_beta, min_periods)
    return s

def calculate_beta():
    args_list=[]
    for freq in ['D','M']:
        eret = read_unfiltered('stockEret' + freq)  # TODO: filtered or unfiltered?
        rp = read_unfiltered('rp' + freq)
        comb = pd.concat([rp, eret], axis=1)
        if freq=='D':
            windows=[20,60,120,240,480]
        else:
            windows=[12,24,40,60]
        for w in windows:
            args_list.append((freq,comb,w))

    ss=multiprocessing.Pool(4).map(task,args_list)
    df=pd.concat(ss,axis=1,keys=['{}{}'.format(a[0],a[2]) for a in args_list])
    df=df.unstack('sid').resample('M').last().stack() # convert to monthly
    save(df,'beta',sort_axis=False)



def _cal_beta1(df):
    # first column is the market
    X = df.values[:, [0]]
    # prepend a column of ones for the intercept
    X = np.concatenate([np.ones_like(X), X], axis=1)
    # matrix algebra
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values[:, 1:])
    return pd.Series(b[1],index= df.columns[1:], name='beta')

class Arg:
    def __init__(self,df,func,w,freq):
        self.df=df
        self.func=func
        self.w=w
        self.freq=freq

def task1(arg):
    s=my_rolling_apply(arg.df,arg.func,arg.w).stack()
    return s

def get_arg_list():
    arg_list=[]
    for freq in ['D', 'M']:
        eret = read_unfiltered('stockEret' + freq)  # TODO: filtered or unfiltered?
        rp = read_unfiltered('rp' + freq)
        comb = pd.concat([rp, eret], axis=1)
        if freq == 'D':
            windows = [20, 60, 120, 240, 480]
        else:
            windows = [12, 24, 40, 60]
        for w in windows:
            arg_list.append(Arg(comb,_cal_beta1,w,freq))
    return arg_list

def calculate_beta1():
    arg_list=get_arg_list()
    ss=multiprocessing.Pool(2).map(task1,arg_list)
    df=pd.concat(ss,axis=1,keys=['{}{}'.format(arg.freq,arg.w) for arg in arg_list])
    df=df.unstack('sid').resample('M').last().stack() # convert to monthly
    save(df,'beta',sort_axis=False)

if __name__ == '__main__':
    calculate_beta1()