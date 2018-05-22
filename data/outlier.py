# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-19  15:01
# NAME:assetPricing2-outlier.py
from config import OUTLIER_PATH
from data.check import MyError
from pylab import savefig
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

#TODO: detect outliers in src
#TODO: detect outliers in indicators
#TODO: detect outliers in factors

def mad_based_outlier(points, thresh=5): #TODO: thresh
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """

    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def percentile_based_outlier(data, threshold=99): #TODO: thresh?
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100.0 - diff])
    return (data < minval) | (data > maxval)



def _for_2d(df,fn):
    if df.columns.name=='sid':
        # stack dimension reduction from 2d (panel) to 1d (time series)
        df = df.dropna(axis=0, how='all')
        fig, axes = plt.subplots(nrows=2,figsize=(20,12))
        outs=df.apply(lambda s:mad_based_outlier(s.dropna().values).sum(),axis=1)
        ratios=[r/l for r,l in zip(outs,[s.dropna().shape[0] for _,s in df.iterrows()])]
        axes[0].plot(outs.index,ratios)
        axes[0].set_title('ratio of outliers')

        dropname=df.columns.name
        df=df.stack()
        df=df.dropna()
        df=df.reset_index(dropname,drop=True)
        axes[1].plot(df.index,df.values,'bo',markersize=3)
        axes[1].set_title("scatter")

        fig.suptitle('outliers for 2d')
        savefig(os.path.join(OUTLIER_PATH, fn + '.png'))
    elif df.columns.name=='type':
        for col,s in df.iteritems():
            _for_1d(s,'{}_{}'.format(fn,col))

def _for_1d(s,fn):
    s=s.dropna()
    fig,axes=plt.subplots(nrows=2,figsize=(20,12))
    for ax,func,type in zip(axes,[mad_based_outlier,percentile_based_outlier],['mad_badsed','percentile_based']):
        outliers=s[func(s)]
        ax.plot(s.index,s,'o',markersize=1)
        ax.plot(outliers.index,outliers,'ro',markersize=3)
        ax.set_title('{};outliers:{}'.format(type,len(outliers)))
    fig.suptitle('sample:{},mean:{:.6f},median:{:.6f}'.format(len(s),np.mean(s),np.median(s)), size=14)
    savefig(os.path.join(OUTLIER_PATH, fn + '.png'))

def _for_2d_multiIndex(multiDf,fn):
    for col,s in multiDf.iteritems():
        df=s.unstack(level='sid')
        _for_2d(df,'{}_{}'.format(fn,col))

def detect_outliers(x, fn):
    '''
    detect outliers

    :param x:Series or DataFrame
    :param fn:
    :param by:
    :return:
    '''

    if x.ndim==1:
        _for_1d(x,fn)
    elif x.ndim==2:
        if isinstance(x.index, pd.MultiIndex):
            _for_2d_multiIndex(x, fn)
        else:
            _for_2d(x,fn)



#---------------------handle outliers-------------------------------
def delete_outliers(x,method,thresh,pooled=True):#TODO: thresh?
    '''
    delete outliers

    Example:
        liquidity=load_data('liquidity')
        ps1=liquidity['ps1'].unstack()
        ps1_mad=delete_outliers(ps1, method='mad', thresh=6,pooled=False)
        ps1_percentile=delete_outliers(ps1, method='percentile', thresh=99,pooled=False)
        ps1_pooled=delete_outliers(ps1,method='percentile',thresh=99,pooled=True)
        ps1_pooled_mad=delete_outliers(ps1,method='mad',thresh=6,pooled=True)

    :param x:
    :param method:{'mad','percentile'}
    :return: the same data structure as x,but the shape may be different.
    '''
    def _for_series(s):
        s=s.dropna()
        if s.shape[0]>0:
            if method=='mad':
                s=s[~mad_based_outlier(s,thresh)]
            elif method=='percentile':
                s=s[~percentile_based_outlier(s,thresh)]
            else:
                raise MyError('no method named "{}"'.format(method))
            return s
        else:
            '''
            For DataFrame.apply(func),the func must return the objects shared the same type.
            For s.shape[0]==0,if we do not return a empty Series,it will return a None,which
            is different with scenery when df.shape[0]>0,which return a Series.
            '''
            return pd.Series()

    def _for_2d_singleIndexed(df):
        if pooled:
            '''
            treat all the element in a df equally,rather than handle them row-by-row or col-by-col
            '''
            return _for_series(df.stack()).unstack()

        if df.columns.name=='sid':# row by row
            '''
            For df like stockCloseD,we handle outliers row by row.
            That is,the outliers are identified and deleted based on all the available 
            close prices at each time point.
            '''
            return df.apply(_for_series,axis=1)
        elif df.columns.name=='type':# column by column
            '''
            For df like ff3D,we handle outliers column by column,that is,every time,we select 
            all the history of a indicator,such  as 'rp',and then identify and delete the outliers
            based on this time series
            '''
            return df.apply(_for_series,axis=0)

    def _for_2d_multiIndexed(multiDf):
        return multiDf.apply(lambda s:_for_2d_singleIndexed(s.unstack('sid')).stack())

    result=None
    if x.ndim==1:
        result=_for_series(x)
        result.name=x.name
    elif x.ndim==2:
        if isinstance(x.index,pd.MultiIndex):
            result=_for_2d_multiIndexed(x)
        else:
            result=_for_2d_singleIndexed(x)
        # DataFrame.apply will  result in the missing of columns.name
        result.columns.name=x.columns.name

    return result



#TODO: save outliers to re-analyse

#TODO: add description function

