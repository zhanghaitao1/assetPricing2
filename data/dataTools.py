# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-21  19:10
# NAME:assetPricing2-dataTools.py
import pandas as pd
import os
import numpy as np

from config import DATA_SRC, CSV_PATH, PKL_PATH
from data.check import check
from data.outlier import detect_outliers


def read_gta(tbname, *args, **kwargs):
    df=pd.read_csv(os.path.join(DATA_SRC,tbname+'.csv'),*args,**kwargs)
    return df

def read_df_from_gta(tbname, varname, indname, colname):
    table=read_gta(tbname)
    df=pd.pivot_table(table,varname,indname,colname)
    return df

def unify(x):
    if x.ndim==1:
        x=x.sort_index()
        return x
    elif x.ndim==2:
        x=x.sort_index(axis=0)
        x=x.sort_index(axis=1)
        return x

def read_raw(tbname,type='pkl',*args,**kwargs):
    if type=='pkl':
        df = pd.read_pickle(os.path.join(PKL_PATH, tbname + '.pkl'))
    else:
        df = pd.read_csv(os.path.join(CSV_PATH, tbname + '.csv'),*args,**kwargs)
        # TODO: datetime,axis dtypes
    return df

def save(x, name,outliers=True):
    '''
    Since some information about DataFrame will be missing,such as the dtype,columns.name,
    will save them as pkl.
    :param x:
    :param name:
    :param outliers:
    :return:
    '''
    x=unify(x)
    check(x,name)
    if outliers:
        detect_outliers(x,name)

    if x.ndim==1:
        x.to_frame().to_csv(os.path.join(CSV_PATH, name + '.csv'))
    else:
        x.to_csv(os.path.join(CSV_PATH, name + '.csv'))

    x.to_pickle(os.path.join(PKL_PATH, name + '.pkl'))
    

def detect_freq(axis):
    days=(axis[1:]-axis[:-1]).days.values
    # avg=np.mean(days)
    # max=np.max(days)
    min=np.min(days)
    if min==1:
        return 'D'
    elif min>=28:
        return 'M'
    else:
        raise ValueError
