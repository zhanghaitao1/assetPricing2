# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-21  19:10
# NAME:assetPricing2-dataTools.py
import pandas as pd
import os
import numpy as np

from config import CSV_PATH, PKL_PATH, PKL_FILTERED_PATH
from data.base import MyError
from data.check import check_data_structure, check_axis_order, check_axis_info
from data.outlier import detect_outliers
from zht.data.gta.api import read_gta


def read_df_from_gta(tbname, varname, indname, colname):
    table=read_gta(tbname)
    df=pd.pivot_table(table,varname,indname,colname)
    return df


def read_unfiltered(tbname, suffix='pkl', *args, **kwargs):
    if suffix== 'pkl':
        df = pd.read_pickle(os.path.join(PKL_PATH, tbname + '.pkl'))
    else:
        df = pd.read_csv(os.path.join(CSV_PATH, tbname + '.csv'),*args,**kwargs)
        # TODO: datetime,axis dtypes
    return df

def save(x, name, data_structure=True,axis_info=True,sort_axis=True):
    '''
        before saving the data as pkl,we can check whether the data accords with
    our standard in respect of data structure,axis info,and the order of axis.

    :param x:Series or DataFrame
    :param name: The name to save,without suffix
    :param data_structure:
    :param axis_info:
    :param sort_axis:
    :return:
    '''
    if data_structure:
        check_data_structure(x)

    if axis_info:
        check_axis_info(x,name)

    if sort_axis:
        x=check_axis_order(x)

    # if outliers:
    #     detect_outliers(x,name)

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
    _min=np.min(days)
    if _min==1:
        return 'D'
    elif _min>=28:
        return 'M'
    else:
        raise ValueError

def load_data(name):
    fns1=os.listdir(PKL_FILTERED_PATH)
    fns2=os.listdir(PKL_PATH)
    if name+'.pkl' in fns1:
        x=pd.read_pickle(os.path.join(PKL_FILTERED_PATH, name + '.pkl'))
    elif name+'.pkl' in fns2:
        x=read_unfiltered(name)
    else:
        raise MyError('There is no such data named "{}.pkl" in the repository!'.format(name))
    return x

def save_to_filtered(x, name):

    # save csv
    if x.ndim==1:
        x.to_frame().to_csv(os.path.join(PKL_FILTERED_PATH, name + '.csv'))
    else:
        x.to_csv(os.path.join(CSV_PATH, name + '.csv'))

    x.to_pickle(os.path.join(PKL_FILTERED_PATH, name + '.pkl'))


