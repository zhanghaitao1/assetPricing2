# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-19  13:44
# NAME:assetPricing2-din.py

import pandas as pd
import os
import pickle

from config import DATA_SRC,CSV_PATH,PKL_PATH
from data.check import check_df, MyError,check
from data.outlier import detect_outliers,  detect_outliers
from pandas.tseries.offsets import MonthEnd
from data.outlier import _for_1d
from pylab import savefig

def read_src(tbname):
    df=pd.read_csv(os.path.join(DATA_SRC,tbname+'.csv'))
    return df

def get_df(tbname, varname, indname, colname):
    table=read_src(tbname)
    df=pd.pivot_table(table,varname,indname,colname)
    return df

def read_csv(tbname, index_col):
    df=pd.read_csv(os.path.join(CSV_PATH,tbname+'.csv'),index_col=index_col)
    # TODO: datetime,axis dtypes
    return df

def read_pkl(tbname):
    df=pd.read_pickle(os.path.join(PKL_PATH,tbname+'.pkl'))
    return df


def series_or_df(x):
    if x.ndim==2 and x.shape[1]==1:
        raise MyError('For DataFrame with only one column,we should convert them to Series!')


def unify(x):
    if x.ndim==1:
        x=x.sort_index()
        return x
    elif x.ndim==2:
        x=x.sort_index(axis=0)
        x=x.sort_index(axis=1)
        return x


def save(x, name):
    '''
    Since some information about DataFrame will be missing,such as the dtype,columns.name,
    will save them as pkl.
    :param x:
    :param name:
    :return:
    '''
    check(x,name)
    detect_outliers(x,name)
    x=unify(x)

    if x.ndim==1:
        x.to_frame().to_csv(os.path.join(CSV_PATH, name + '.csv'))
    else:
        x.to_csv(os.path.join(CSV_PATH, name + '.csv'))

    x.to_pickle(os.path.join(PKL_PATH, name + '.pkl'))

def get_stockRetD():
    # get stock daily stock return
    tbname='TRD_Dalyr'
    varname='Dretwd'#考虑现金红利再投资的收益
    indname='Trddt'
    colname='Stkcd'
    df=get_df(tbname, varname, indname, colname)

    df.index.name='t'
    df.index=pd.to_datetime(df.index) #TODO: dayend?
    df.columns.name='sid'
    df.columns=df.columns.astype(str)

    save(df, 'stockRetD')

def get_stockCloseD():
    #get daily stock close price
    tbname='TRD_Dalyr'
    varname='Clsprc'
    indname='Trddt'
    colname='Stkcd'
    df=get_df(tbname,varname,indname,colname)

    df.index.name='t'
    df.index=pd.to_datetime(df.index)
    df.columns.name='sid'
    df.columns=df.columns.astype(str)

    df=df.sort_index(axis=0)
    df=df.sort_index(axis=1)

    save(df, 'stockCloseD')


def get_mktRetD():
    # get daily market return

    tbname = 'TRD_Cndalym'
    indVar = 'Trddt'
    targetVar = 'Cdretwdos'  # 考虑现金红利再投资的综合日市场回报率(流通市值加权平均法)
    df=read_src(tbname)


    condition1=df['Markettype']==21 # 21=综合A股和创业板
    df=df[condition1]

    df = df.set_index(indVar)
    df.index.name='t'
    df.index = pd.to_datetime(df.index)

    s = df[targetVar]
    s.name='mktRetD' #TODO: put this line into check function or unify function?
    save(s, 'mktRetD')










def get_stockRetM():
    '''
    monthly stock return with dividend
    '''
    tbname = 'TRD_Mnth'
    varname='Mretwd'#考虑现金红利再投资的收益
    indname='Trdmnt'
    colname='Stkcd'

    df=get_df(tbname, varname, indname, colname)

    #TODO: identify the axis and convert the axis automatically
    df.index.name='t'
    df.columns.name='sid'
    df.index=pd.to_datetime(df.index)+MonthEnd(0)
    df.columns=df.columns.astype(str)

    save(df, 'stockRetM')
