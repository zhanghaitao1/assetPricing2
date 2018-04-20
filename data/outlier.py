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

# def detect_outliers1(s,thresh=2):
#     s=s.dropna()
#     n=s.shape[0]
#     avg=s.mean() #TODO: how about median?
#     std=s.std()
#     up=(s>avg+thresh*std).sum()*100/n
#     down=(s<avg-thresh*std).sum()*100/n
#     return n,avg,s.max(),s.min(),up,down

# def show_outliers(x, name=None):
#     '''
# 
#     :param x:DataFrame of Series
#     :param name:
#     :return:
#     '''
# 
#     def _for_series(s, name):
#         avg = s.mean()
#         std = s.std()
#         length = s.shape[0]
# 
#         plt.figure()
#         plt.plot(s.index, s, 'ro', markersize=1)
#         plt.plot(s.index, [0.0] * length, 'b--', linewidth=0.5)
# 
#         plt.plot(s.index, [avg] * length, 'r--', linewidth=1)
# 
#         plt.plot(s.index, [avg - 2 * std] * length, 'r--', linewidth=0.5)
#         plt.plot(s.index, [avg + 2 * std] * length, 'r--', linewidth=0.5)
# 
#         plt.plot(s.index, [avg - 3 * std] * length, 'r--', linewidth=0.5)
#         plt.plot(s.index, [avg + 3 * std] * length, 'r--', linewidth=0.5)
# 
#         plt.plot(s.index, [avg - 4 * std] * length, 'r--', linewidth=0.5)
#         plt.plot(s.index, [avg + 4 * std] * length, 'r--', linewidth=0.5)
# 
#         plt.plot(s.index, [avg - 5 * std] * length, 'r--', linewidth=0.5)
#         plt.plot(s.index, [avg + 5 * std] * length, 'r--', linewidth=0.5)
# 
#         plt.show()
# 
#         # savefig(r'e:\a\{}.png'.format(name))
# 
#     if x.ndim == 1:
#         _for_series(x, name)
#     else:
#         for colname, s in x.iteritems():
#             _for_series(s, colname)


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


def _for_2d(df,by='rbr'):
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
        axes[1].plot(df.index,df.values,'ro',markersize=3)
        axes[1].set_title("scatter")
        
        fig.suptitle('outliers for 2d')
        
    elif by=='cbc':
        pass
    else:
        raise MyError('by should be "rbr" or "cbc"!')

def _for_1d(x):
    x=x.dropna().values
    fig, axes = plt.subplots(nrows=2,figsize=(20,12))
    kwargs = dict(y=0.95, x=0.05, ha='left', va='top')
    for ax, func ,type in zip(axes, [percentile_based_outlier, mad_based_outlier],['percentile_based','mad_based']):
        sns.distplot(x, ax=ax, rug=True, hist=False)
        outliers = x[func(x)]
        ax.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)
        ax.set_title('{};outliers:{}'.format(type,len(outliers)),kwargs)

    fig.suptitle('sample:{},mean:{:.6f},median:{:.6f}'.format(len(x),np.mean(x),np.median(x)), size=14)



def detect_outliers(x, fn, by='rbr'):
    if x.ndim==1:
        _for_1d(x)
    elif x.ndim==2:
        _for_2d(x,by)

    savefig(os.path.join(OUTLIER_PATH, fn + '.png'))


#TODO: save outliers to re-analyse


