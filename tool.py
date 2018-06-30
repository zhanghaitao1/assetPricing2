# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-22  15:03
# NAME:assetPricing2-dataTools.py
import multiprocessing
from functools import partial

from config import WINSORIZE_LIMITS
from dout import *
import numpy as np
import time
import scipy
import pandas as pd
from pandas.io.formats.format import format_percentiles
import statsmodels.formula.api as sm
from numpy.linalg import inv

from zht.utils import mathu
import pylab as plt
from pylab import savefig

#TODO: multiIndex span on the index and groupby


def my_rolling(func, x, window, freq, min_periods, *args, **kwargs):
    '''
    x can be series or Dataframe

    :param func:function running on a series and return a scalar
    :param x:
    :param window:
    :param freq:
    :param min_periods:
    :param args:
    :param kwargs:
    :return:
    '''
    def _rolling_1d(func,s,window,freq,min_periods,*args,**kwargs):
        '''
        s can be singleIndex series or multiIndex Series
        '''
        if isinstance(s.index,pd.MultiIndex):
            indnames=list(s.index.names)
            indnames.remove('t')
            s=s.reset_index(indnames,drop=True)

        if freq == 'M':
            days = s.index.get_level_values('t').unique()
            months = days[days.is_month_end] #TODO:wrong!!!!!!!!

            values = []
            for month in months:
                subx = s.loc[:month].last(window).dropna()
                if subx.shape[0] > min_periods:
                    values.append(func(subx, *args, **kwargs))
                else:
                    values.append(np.nan)
            return pd.Series(values, index=months)
        else:
            print('This function only support freq=="M" model')

    if x.ndim==1:
        return _rolling_1d(func,x,window,freq,min_periods,*args,**kwargs)
    elif x.ndim==2:
        return x.apply(lambda s:_rolling_1d(func,s,window,freq,min_periods,*args,**kwargs))



def group_rolling(func,x,groupby,window,freq,min_periods,*args,**kwargs):
    '''
    x must be a Series or DataFrame with multiIndex

    :param func:function running on a series and return a scalar
    :param x:
    :param groupon:
    :param freq:
    :param min_periods:
    :param args:
    :param kwargs:
    :return:
    '''
    def _group_rolling_1d(func, Series, groupby, window,freq,min_periods, *args, **kwargs):

        result=Series.groupby(groupby).apply(
            lambda s:my_rolling(func, s, window, freq, min_periods, *args, **kwargs)
        )
        return result

    if x.ndim==1:
        return _group_rolling_1d(func, x, groupby, window,freq,min_periods, *args, **kwargs)
    elif x.ndim==2:
        return x.apply(lambda s:_group_rolling_1d(func, s, groupby, window,freq,min_periods, *args, **kwargs))


#TODO: use s.rolling to upgrade this fuction as 4momentum.py
def _rolling_for_series(x, months, history, thresh, type_func):
    '''
    calculate the indicator for one stock,and get a time series

    :param x:series or pandas DataFrame
    :param months:list,contain the months to calculate the indicators
    :param history:history length,such as 12M
    :param thresh:the mimium required observe number
    :param type_func:the function name from one of [_skew,_coskew,_idioskew]
    :return:time series
    '''
    # sid=x.index.get_level_values('sid')[0]
    x=x.reset_index('sid',drop=True)
    values=[]
    for month in months:
        #TODO: we use calendar month rather than the absolute 30 days' window
        subx=x.loc[:month].last(history)
        subx=subx.dropna()
        if subx.shape[0]>thresh:
            values.append(type_func(subx))
        else:
            values.append(np.nan)
    return pd.Series(values,index=months)

def groupby_rolling(multiIndDF, prefix, dict, type_func):
    values = []
    names = []
    #TODO:why not use map or other higher-order function

    for history, thresh in dict.items():
        days = multiIndDF.index.get_level_values('t').unique()
        # or (days+MonthEnd(0)).unique()
        months=pd.date_range(start=days[0],end=days[-1],freq='M')
        value = multiIndDF.groupby('sid').apply(
            lambda df: _rolling_for_series(df, months, history, thresh, type_func))
        values.append(value.T)
        names.append(prefix + '_' + history)
        print(history,thresh)
    result = pd.concat(values, axis=0, keys=names)
    return result

def groupby_rolling1(multiIndDF,func,history,thresh):
    days = multiIndDF.index.get_level_values('t').unique()
    months = pd.date_range(start=days[0], end=days[-1], freq='M')
    value = multiIndDF.groupby('sid').apply(
        lambda df: _rolling_for_series(df, months, history, thresh, func))
    value.columns.name='t'
    return value


#TODO: use closures and decorator to handle this problem
#TODO:upgrade this funcntion use rolling,rolling_apply
def monthly_cal(comb, prefix, dict, type_func, fn):
    result=groupby_rolling(comb,prefix,dict,type_func)
    result.to_csv(os.path.join(DATA_PATH,fn+'.csv'))
    return result

def apply_col_by_col(func):
    '''
        A decorator to convert a function acting on series to a function acting
    on DataFrame,the augmented function will run column by column on the DataFrame

    :param func:
    :return:
    '''
    def augmented(x,*args,**kwargs):
        if x.ndim == 1:
            return func(x,*args,**kwargs)
        elif x.ndim == 2:
            return x.apply(lambda s:func(s,*args,**kwargs)) #TODO: how about apply row by row?

    return augmented


def grouping(x,q,labels,axis=0,thresh=None):
    '''
    sort and name for series or dataframe,for dataframe,axis is required with 0 denoting row-by-row
    and 1 denoting col-by-col
    :param x:
    :param q:
    :param labels:
    :param axis:
    :param thresh:
    :return:
    '''

    def _grouping_1d(series,q,labels,thresh=None):
        if thresh==None:
            thresh=q*10 #TODO:
        series=series.dropna()
        if series.shape[0]>thresh:
            return pd.qcut(series,q,labels)

    if x.ndim==1:
        return _grouping_1d(x,q,labels,thresh)
    else:
        if axis==1:
            return x.apply(lambda s:_grouping_1d(s,q,labels,thresh))
        elif axis==0:
            return x.T.apply(lambda s:_grouping_1d(s,q,labels,thresh))


def assign_port_id(s, q, labels=None, thresh=None):
    '''
    this function will first dropna and then asign porfolio id.

    :param s: Series
    :param q:  integer or array of quantiles Number of quantiles. 10 for deciles,
     4 for quartiles, etc. Alternately array of quantiles,
     e.g. [0, .25, .5, .75, 1.] for quartiles
    :param labels:
    :param thresh:
    :return:
    '''
    s = s.dropna()
    if thresh is None:
        if isinstance(q,int):
            thresh = q * 10  # TODO: thresh self.q*10？
        elif isinstance(q,(list,tuple)):
            thresh=50

    if not labels:
        if isinstance(q,int):
            labels=range(1,q+1)
        elif isinstance(q,(list,tuple)):
            labels=range(1,len(q))

    if s.shape[0] > thresh:
        result = pd.qcut(s, q, labels)
        return result
    else:
        return pd.Series(index=s.index)

def monitor(func):

    def wrapper(*args,**kwargs):
        print('{}   starting -> {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'),func.__name__))
        return func(*args,**kwargs)

    return wrapper


def my_average(df,vname,wname=None):
    '''
    calculate average,allow np.nan in df
    This function intensify the np.average by allowing np.nan

    :param df:DataFrame
    :param vname:col name of the target value
    :param wname:col name of the weights
    :return:scalar
    '''
    if wname is None:
        return df[vname].mean()
    else:
        df=df.dropna(subset=[vname,wname])
        if df.shape[0]>0:
            return np.average(df[vname],weights=df[wname])

def GRS_test(factor,resid,alpha):
    '''
        returns the GRS test statistic and its corresponding p-value proposed in
    Gibbons,Ross,Shanken (1989),to test the hypothesis:alpha1=alpha2=...=alphaN=0.
    That is if the alphas from time series Regression on N Test Assets are
    cummulatively zero.

    Reference:
        Gibbons, M.R., Ross, S., and Shanken, J. (1989). A Test of the Efficiency of a Given Portfolio. Econometrica 57, 1121–1152.

    For matrix operation in numpy ,notice that matrix is different with array
    in Numpy,refer to https://stackoverflow.com/questions/4151128/what-are-the-differences-between-numpy-arrays-and-matrices-which-one-should-i-u

    Args:
        factor:TxL Matrix of factor returns (the "Matrix" means numpy.matrixlib.defmatrix.matrix)
        resid: TxN Matrix of residuals from TS-regression
        alpha: Nx1 Matrix of intercepts from TS-regression

    details:
        N:the number of assets (or the number of regressions)
        T:the number of the observations
        L:the number of the factors

    Returns:
        GRS[0,0]:1x1 scalar of GRS Test-value
        pvalue[0,0]:1x1 scalar of P-value from an F-Distribution

    '''
    T, N = resid.shape
    L = factor.shape[1]

    mu_mean=factor.mean(0)
    cov_e=np.cov(resid.T)
    # when there is only factor,np.cov(factor.T) will return a one
    # dimensional array,which does not support operator "@",so we use .reshape
    # to convert it into a 2 dimendional ndarray
    cov_f=np.cov(factor.T).reshape((L,L))

    # matrix operation with np.ndarray
    GRS=(T/N)*((T-N-L)/(T-L-1)) * (alpha.T @ inv(cov_e) @ alpha)/(1+mu_mean.T @ inv(cov_f) @ mu_mean)
    GRS=GRS[0,0]
    # for GRS mentioned in the "asset pricing,2015" of Cochrane the equation is as the following
    # GRS1=(T-N-L)/N * (alpha.T @ inv(cov_e) @ alpha)/(1+mu_mean.T @ inv(cov_f) @ mu_mean)
    GRSp=scipy.stats.f.sf(GRS,N,(T-N-L))
    # critical = scipy.stats.f.isf(0.05, N, T - N - L)
    return GRS,GRSp


def summary_statistics(data,percentiles=(0.05, 0.25, 0.5, 0.75, 0.95),axis=1):
    '''
    calculate the summary statistics.
    Args:
        data: pandas Series or pandas DataFrame.
        axis: 0 or 1
            If 0,take out a column each time and summary this column along the index.
            That is,summarize column by column.
    Returns:series or dataframe.

    References:
        pandas.DataFrame.describe()
    '''
    #TODO: upgrade this function,refer to https://pandas.pydata.org/pandas-docs/stable/basics.html#custom-describe

    def describe_1d(s, pcts):
        stat_index=(['mean','std','skew','kurt','min'] +
                    format_percentiles(pcts) + ['max', 'n'])
        d=([s.mean(), s.std(), s.skew(), s.kurt(), s.min()] +
           [s.quantile(x) for x in pcts] + [s.max(), s.count()])
        return pd.Series(d, index=stat_index, name=s.name)

    if data.ndim==1:
        return describe_1d(data,percentiles)
    else:
        ss=data.apply(partial(describe_1d,pcts=percentiles),axis=axis)
        return ss

def cal_corr(df,method='pearson',winsorize=False):
    '''
    compare with the method in dataframe,this function will automatically winsorize the values
    before calculating Pearson correlations.

    Args:
        df:
        method: {'pearson','spearman'},'pearson' denotes Pearson product-moment correlation
            'spearman' denotes the Spearman rank correlation.Values are winsorized prior to
            calculate the Pearson product-moment correlation.When calculating the Spearman
            rank correlation,the data should not be winsorized.

    Returns:

    '''
    if winsorize==True:
        df = mathu.winsorize(df, WINSORIZE_LIMITS, axis = 0) #TODO: winsorize limit

    if method=='pearson':
        return df.corr(method='pearson')
    elif method=='spearman':
        return df.corr(method='spearman')

#persistance
def cal_persistence(df,offsets):
    '''
    calculate cross-sectional Pearson product-moment correlations
    between indicator mesured in time t with indicator measured in
    t+offset.

    Returns:

    References:
        chapter 4.1.1 of (Bali, Turan G., Robert F. Engle, and Scott Murray. Empirical Asset Pricing:
        The Cross Section of Stock Returns. Hoboken, New Jersey: Wiley, 2016.)

    '''
    def _cross_persistence(df,offset):
        inds=df.index.tolist()
        series = pd.Series(name=offset)
        for i in range(len(inds[:-offset])):
            corrdf=df.iloc[[i,i+offset],:].T
            corr=corrdf.corr()
            series[inds[i]]=corr.iloc[0,1]
        return series

    lcorr = [_cross_persistence(df,offset) for offset in offsets]
    persistence = pd.concat(lcorr, axis=1)
    avg = persistence.mean()
    return avg

def cal_breakPoints(df,q):
    '''
    If the q is an integer,the function will return the breakpoints
    to break the data into q portfolios,that is it will create q+1
    breakpoints.If the q is a list of quantiles,then the function will
    create breakpoints according the given quantiles.

    Args:
        df: dataframe
        q: integer or array of quantiles

    Returns:

    '''

    if isinstance(q,int):
        points=np.linspace(0,1.0,q+1)
    else:
        points=q
    def _for_series(s,points):
        return pd.Series([s.quantile(bp) for bp in points],index=format_percentiles(points))

    breakpoints=df.apply(lambda s:_for_series(s,points),axis=1)

    return breakpoints


def count_groups(df,q):
    '''
    Count the number of stocks for each group,given the quantiles or
    the number of groups

    Args:
        df:
        q: integer or array of quantiles
            Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately
            array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles
    Returns:

    '''
    if isinstance(q,int):
        lbs=['g'+str(i) for i in range(1,q+1)]
    else:
        lbs=['g'+str(b) for b in q]

    def _for_series(s,lbs):
        return pd.qcut(s,q,labels=lbs).value_counts()

    count=df.apply(lambda s:_for_series(s,lbs),axis=1)
    return count

def famaMacBeth(formula, time_label, df, lags=5):
    '''
    Fama-MacBeth regression with Newey-West correction.

        The only input required for the Newey and West (1987) adjustment is
    the number of lags to use when performing the adjustment. As discussed
    in Newey and West (1994), the choice of lags is arbitrary.Frequently,
    econometrics software sets the number of lags to 4(T/100)a, where T is
    the number of periods in the time series, a = 2/9 when using the Bartlett
    kernel, and a = 4/25 when using the quadratic spectral kernel to calculate
    the autocorrelation and heteroscedasticity-adjusted standard errors.A
    large proportion of empirical asset pricing studies use monthly samples
    covering the period from 1963 through the present (2012, or T = 600
    months for the data used in this book). Plugging in the value T = 600
    and taking a to be either 2/9 or 4/25 results in a value between five
    and six. Most studies, therefore, choose six as the number of lags.

    Args:
        formula:string like 'y ~ x'
        time_label:string
            choosen from the df.columns,it used to indicate which column is time
        df:
        lags:int
            number of lags to use when performing the adjustment.
            If lags==0,it will forgo the Newey-West correction.


    References:
        1.http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/se_programming.htm

        2.Fama, Eugene F., and James D. MacBeth. “Risk, Return, and Equilibrium:
        Empirical Tests.” Journal of Political Economy 81, no. 3 (May 1, 1973):
         607–36. https://doi.org/10.1086/260061.

        3.Bali, Turan G., Robert F. Engle, and Scott Murray. Empirical Asset
        Pricing: The Cross Section of Stock Returns. Hoboken, New Jersey: Wiley, 2016.

    Examples:
        from urllib.request import urlopen
        filehandle = urlopen('http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.txt')
        df = pd.read_table(filehandle, names=['firmid','year','x','y'],
                  delim_whitespace=True)

        famaMacBeth('y ~ x','year',df,lags=3)

                   coef    stderr     tvalue        pvalue stars
        Intercept  0.031278  0.021295   1.468810  1.759490e-01
        x          1.035586  0.025883  40.010988  1.893637e-11   ***

    '''

    res = df.groupby(time_label,sort=True).apply(lambda x: sm.ols(
        formula, data=x).fit())
    if time_label in df.columns:
        groups=df[time_label].unique()
    else:# The time label comes from the index
        groups=df.index.get_level_values(time_label).unique()

    p=pd.DataFrame([x.params for x in res],index=groups)
    N=np.mean([x.nobs for x in res])
    adj_r2=np.mean([x.rsquared_adj for x in res])

    means = {}
    params_labels = res.iloc[0].params.index

    for x in p.columns:
        if lags is 0:
            means[x] = sm.ols(formula=x + ' ~ 1',
                              data=p[[x]]).fit(use_t=True)
        else:
            means[x] = sm.ols(formula=x + ' ~ 1',
                              data=p[[x]]).fit(cov_type='HAC',
                                cov_kwds={'maxlags': lags},
                                use_t=True)

    params = []
    stderrs = []
    tvalues = []
    pvalues = []
    for x in params_labels:
        params.append(means[x].params['Intercept'])
        stderrs.append(means[x].bse['Intercept'])
        tvalues.append(means[x].tvalues['Intercept'])
        pvalues.append(means[x].pvalues['Intercept'])

    result = pd.DataFrame([params, stderrs, tvalues, pvalues]).T
    result.index = params_labels
    result.columns = ['coef', 'stderr', 'tvalue', 'pvalue']
    result['stars'] = ''
    result.loc[result.pvalue < 0.1, 'stars'] = '*'
    result.loc[result.pvalue < 0.05, 'stars'] = '**'
    result.loc[result.pvalue < 0.01, 'stars'] = '***'

    return result,adj_r2,N,p

def newey_west(formula,df,lags=5):
    #TODO: other t values such as bootstrapped standard errors of Murphy and Topel, “Estimation and Inference in Two-Step Econometric Models.”
    #TODO: Shanken corrected t-statistics
    reg=sm.ols(formula,df).fit(cov_type='HAC',
                                  cov_kwds={'maxlags':lags},
                                  use_t=True)
    return pd.DataFrame([reg.params,reg.tvalues],index=['coef','t'],
                        columns=reg.params.index)

def get_riskAdjusted_alpha_tvalue(s, bench=None):
    s.name = 'y'
    s=s.to_frame()
    if bench is not None:
        df=pd.concat([s,bench],axis=1)
        formula='y ~ {}'.format(' + '.join(bench.columns.tolist()))
        nw=newey_west(formula,df)
        return nw['Intercept']['t']
    else:
        formula='y ~ 1'
        nw = newey_west(formula, s)
        return nw['Intercept']['t']

def correlation_mixed(multiDf):
    indicators=multiDf.columns.tolist()

    def _spearman(df):
        df = df.dropna()
        if df.shape[0] > 10:  # TODO:thresh to choose
            return cal_corr(df, 'spearman', winsorize=False)

    def _pearson(df):
        df = df.dropna()
        if df.shape[0] > 10:  # TODO: min_samples
            return cal_corr(df, 'pearson', winsorize=True)

    corrs = multiDf.groupby('t').apply(_spearman)
    corrp = multiDf.groupby('t').apply(_pearson)

    corrsAvg = corrs.groupby(level=1).mean().reindex(index=indicators,
                                                     columns=indicators)
    corrpAvg = corrp.groupby(level=1).mean().reindex(index=indicators,
                                                     columns=indicators)

    corr1 = np.tril(corrpAvg.values, k=-1)
    corr2 = np.triu(corrsAvg.values, k=1)

    corr = pd.DataFrame(corr1 + corr2, index=corrpAvg.index,
                        columns=corrpAvg.columns)
    np.fill_diagonal(corr.values, np.NaN)
    return corr

def multi_processing(task,params,pool_size=8,**kwargs):
    pool=multiprocessing.Pool(pool_size)
    local_multiprocessing_func=partial(task,**kwargs)
    result=pool.map(local_multiprocessing_func,params)
    pool.close()
    return result


