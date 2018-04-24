# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-21  19:10
# NAME:assetPricing2-dataTools.py
import pandas as pd
import os
import numpy as np

from config import DATA_SRC, CSV_PATH, PKL_PATH, FILTERED_PATH
from data.base import MyError
from data.check import is_valid
from data.outlier import detect_outliers
from zht.data.gta.api import read_gta


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

def save(x, name, validation=True, outliers=True):
    '''
    Since some information about DataFrame will be missing,such as the dtype,columns.name,
    will save them as pkl.
    :param x:
    :param name:
    :param outliers:
    :return:
    '''
    if validation:
        x=unify(x)
        is_valid(x, name)

    if outliers:
        detect_outliers(x,name)

    # save csv
    if x.ndim==1:
        x.to_frame().to_csv(os.path.join(CSV_PATH, name + '.csv'))
    else:
        x.to_csv(os.path.join(CSV_PATH, name + '.csv'))

    x.to_pickle(os.path.join(PKL_PATH, name + '.pkl'))
    

def detect_freq(axis):
    ts=axis.get_level_values('t').unique()
    days=(ts[1:]-ts[:-1]).days.values
    # avg=np.mean(days)
    # max=np.max(days)
    min=np.min(days)
    if min==1:
        return 'D'
    elif min>=28:
        return 'M'
    else:
        raise ValueError

def load_data(name):
    fns1=os.listdir(FILTERED_PATH)
    fns2=os.listdir(PKL_PATH)
    if name+'.pkl' in fns1:
        x=pd.read_pickle(os.path.join(FILTERED_PATH,name+'.pkl'))
    elif name+'.pkl' in fns2:
        x=read_raw(name)
    else:
        raise MyError('There is no such data named "{}.pkl" in the repository!'.format(name))
    return x

def save_to_filter(x,name):
    x.to_pickle(os.path.join(FILTERED_PATH,name+'.pkl'))
