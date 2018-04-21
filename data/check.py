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
    if delta.days>=28 and axis[1].day<28:
        raise MyError('The frequency seems to be "M",but the date is not the end of month !')

def _check_multiIndex(axis):
    dic={'t':pd.Timestamp,'sid':str}
    names=axis.names
    values=axis[0]
    for n,v in zip(names,values):
        if not isinstance(v,dic[n]):
            raise MyError('The data type of "{}" should be "{}",rather than "{}"!'.format(n,dic[n],type(v)))

    if axis.has_duplicates:
        raise MyError('The axis "{}" has duplicates'.format(axis.name))
    
def _check_singleIndex(axis):
    #for single index
    dic={'sid':str,
         't':pd.Timestamp,
         'type':str}

    # check data type
    if not axis.name:
        raise MyError('axis name is missing !')
    elif axis.name not in dic.keys():
        raise MyError('The axis name is "{}",not included in {}'.format(axis.name,str(dic.keys())))
    elif not isinstance(axis[0],dic[axis.name]):
        raise MyError('The data type of "{}" should be "{}",rather than "{}"!'.format(
            axis.name, dic[axis.name], type(axis[0])))

    # check duplicates
    if axis.has_duplicates:
        raise MyError('The axis "{}" has duplicates'.format(axis.name))

def _check_axis(axis):
    if isinstance(axis,pd.MultiIndex):
        _check_multiIndex(axis)
    else:
        _check_singleIndex(axis)

def check_s_for_saving_name(s, name):
    if not s.name:
        raise MyError('No name for Series')
    elif s.name!=name:
        raise MyError('The file name "{}" to save is different with the name of Series "{}"'.format(name,s.name))

def check_s(s, name):
    _check_axis(s.index)
    check_s_for_saving_name(s, name)

def check_df(df):
    if len(df.columns)<=1:
        raise MyError("For DataFrame with only one column,you'd better convert it to Series !")
    _check_axis(df.index)
    _check_axis(df.columns)

def check(x,name):
    if x.ndim==1:
        check_s(x,name)
    elif x.ndim==2:
        check_df(x)
    else:
        raise NotImplementedError
