# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-19  15:21
# NAME:assetPricing2-check.py

'''
This module is used to check the validality of the Pandas objects from the following aspects:
1.
'''


import pandas as pd
from data.base import MyError




def identify_axis():
    #TODO: machine learning to identify the type and name of axis
    pass

def check_axis_order(x):
    if x.ndim==1:
        x=x.sort_index()
        return x
    elif x.ndim==2:
        x=x.sort_index(axis=0)
        x=x.sort_index(axis=1)
        return x


def check_data_structure(x):
    '''
    All the data structure should belong to the following list:
    1. singleIndexed Series (index.name,series.name is required)
    2. singleIndexed DataFrame with multiple columns(index.name,columns.name is required)
    3. multIndexed DataFrame with multiple columns(index.names,columns.names is required)

    Rules:
    1. If there  is "t" axis,always put it in index.
    2. For multiIndexed DataFrame,if there is 't','sid',they should be put in index,with 't' as level0 and
        'sid' as level1,just like
                            'col1'  'col2'
            't'     'sid'     a1      a2
            '1990'   '1'      a1      a2
            '1990'   '2'      a1      a2
            '1991'   '1'      a1      a2
            '1991'   '2'      a1      a2

    '''
    if x.ndim==1 and isinstance(x.index,pd.MultiIndex):
        raise MyError("Series with MultiIndex is not allowed ! you'd betterconvert it into singleIndexed DataFrame !")
    elif x.ndim==2 and x.shape[1]==1:
        raise MyError("DataFrame with only one column is not allowed,you'd better convert it to Series !")


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

def _check_axis_info(axis):
    if isinstance(axis,pd.MultiIndex):
        _check_multiIndex(axis)
    else:
        _check_singleIndex(axis)

def _check_s_for_saving_name(s, name):
    if not s.name:
        raise MyError('No name for Series')
    elif s.name!=name:
        raise MyError('The file name "{}" to save is different with the name of Series "{}"'.format(name,s.name))

def check_axis_info(x,name):
    if x.ndim==1:
        _check_axis_info(x.index)
        _check_s_for_saving_name(x, name)

    elif x.ndim==2:
        _check_axis_info(x.index)
        _check_axis_info(x.columns)
    else:
        raise NotImplementedError
