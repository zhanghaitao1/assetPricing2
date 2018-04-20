# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-19  15:21
# NAME:assetPricing2-check.py

import pandas as pd

class MyError(Exception):
    '''The name of index is wrong'''
    def __init__(self,msg):
        super().__init__(msg)

def identify_axis(axis):
    #TODO: machine learning to identify the type and name of axis

    pass


def analyse_freqency(axis):
    delta=axis[2]-axis[1]
    if delta.days>=28 and axis[1].day<=28:
        raise MyError('The frequency seems to be "M",but the date is not the end of month !')

def _check_axis(axis):
    #for single index

    validNames=['sid','t']

    #TODO: how about multiIndex DataFrame and Series?
    if not axis.name:
        raise MyError('axis name is missing !')
    elif axis.name not in ['sid','t']:
        raise MyError('The axis name is {},not included in {}'.format(axis.name,str(validNames)))
    elif axis.name=='t':
        if not isinstance(axis[0],pd.Timestamp):
            raise MyError('The data type of "time index" should be pd.Timestamp rather than {}!'.format(type(axis[0])))
        analyse_freqency(axis)
    elif axis.name=='sid':
        if not isinstance(axis[0],str):
            raise MyError('The data type of "sid" should be str rather than {}!'.format(type(axis[0])))
        #TODO:unify the sid add suffix for sid

    if axis.has_duplicates:
        raise MyError('The axis "{}" has duplicates'.format(axis.name))


def check_s_for_saving_name(s, name):
    if not s.name:
        raise MyError('No name for Series')
    elif s.name!=name:
        raise MyError('The file name "{}" to save is different with the name of Series "{}"'.format(name,s.name))

def check_s(s, name):
    _check_axis(s.index)
    check_s_for_saving_name(s, name)

def check_df(df):
    _check_axis(df.index)
    _check_axis(df.columns)

def check(x,name):
    if x.ndim==1:
        check_s(x,name)
    elif x.ndim==2:
        check_df(x)
    else:
        raise NotImplementedError
