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
import seaborn as sns
import os
import pandas as pd

#TODO: detect outliers in src
#TODO: detect outliers in indicators
#TODO: detect outliers in factors

def mad_based_outlier(points, thresh=3.5):
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

def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)




def _for_2d(df,fn,by='rbr'):
    if by=='rbr':
        df = df.dropna(axis=0, how='all')
        fig, axes = plt.subplots(nrows=2,figsize=(20,12))
        result=df.apply(lambda s:mad_based_outlier(s.dropna().values).sum(),axis=1)
        ratios=[r/l for r,l in zip(result,[s.dropna().shape[0] for _,s in df.iterrows()])]
        axes[0].plot(result.index,ratios)
        axes[0].set_title('ratio of outliers')
        
        dropname=df.columns.name
        df=df.stack()
        df=df.dropna()
        df=df.reset_index(dropname,drop=True)
        axes[1].plot(df.index,df.values,'bo',markersize=3)
        axes[1].set_title("scatter")
        
        fig.suptitle('outliers for 2d')
        savefig(os.path.join(OUTLIER_PATH, fn + '.png'))

    elif by=='cbc':
        pass
    else:
        raise MyError('by should be "rbr" or "cbc"!')

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

def detect_outliers(x, fn, by='rbr'):
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
            _for_2d(x,fn,by)


#TODO: save outliers to re-analyse

#TODO: add description function

