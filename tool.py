# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-22  15:03
# NAME:assetPricing2-tool.py
from dataset import DATA
from dout import *
import numpy as np

#TODO:upgrade this funcntion use rolling
from zht.utils import assetPricing


def _for_one_stock(x, months, history, thresh, type_func):
    '''
    calculate the indicator for one stock,and get a time series

    :param x:series or pandas DataFrame
    :param months:list,contain the months to calculate the indicators
    :param history:history length,such as 12M
    :param thresh:the mimium required observe number
    :param type_func:the function name from one of [_skew,_coskew,_idioskew]
    :return:time series
    '''
    sid=x.index.get_level_values('sid')[0]
    x=x.reset_index('sid',drop=True)
    values=[]
    for month in months:
        subx=x.loc[:month].last(history)
        subx=subx.dropna()
        if subx.shape[0]>thresh:
            values.append(type_func(subx))
        else:
            values.append(np.nan)
    print(sid)
    return pd.Series(values,index=months)

def groupby_rolling(multiIndDF, prefix, dict, type_func):
    values = []
    names = []
    for history, thresh in dict.items():
        days = multiIndDF.index.get_level_values('t').unique()
        months = days[days.is_month_end]
        value = multiIndDF.groupby('sid').apply(lambda df: _for_one_stock(df, months, history, thresh, type_func))
        values.append(value.T)
        names.append(prefix + '_' + history)
    result = pd.concat(values, axis=0, keys=names)
    return result

#TODO: use closures and decorator to handle this problem
def monthly_cal(comb, prefix, dict, type_func, fn):
    result=groupby_rolling(comb,prefix,dict,type_func)
    result.to_csv(os.path.join(DATA_PATH,fn+'.csv'))
    return result


def apply_col_by_col(func):
    '''
    a decorator to augment the 1d function ot 1d or 2d function.

    :param func:
    :return:
    '''
    def augmented(x,*args,**kwargs):
        if x.ndim == 1:
            return func(x,*args,**kwargs)
        elif x.ndim == 2:
            return x.apply(lambda s:func(s,*args,**kwargs)) #TODO: how about apply row by row?

    return augmented

@apply_col_by_col
def adjust_with_riskModel(x, riskmodel=None):
    '''
    use risk model to adjust the the alpha,
    the risk model can be None (unadjusted) or one of [capm,ff3,ffc,ff5,hxz4]

    :param x:
    :param riskmodel:
    :return:
    '''
    lags=5
    d={'capm':'rpM',
       'ff3':'ff3M',
       'ffc':'ffcM',
       'ff5':'ff5M',
       'hxz4':'hxz4M'}

    df = pd.DataFrame(x)
    df.columns = ['y']

    if riskmodel in d.keys():
        df=df.join(DATA.by_factor(d[riskmodel]))
        formula='y ~ '+' + '.join(DATA.info[d[riskmodel]])
        nw = assetPricing.newey_west(formula, df, lags)
        return nw['Intercept'].rename(index={'coef': riskmodel+'_alpha',
                                             't': riskmodel+'_alpha_t'})
    else:
        formula='y ~ 1'
        nw = assetPricing.newey_west(formula, df, lags)
        return nw['Intercept'].rename(index={'coef': 'excess return',
                                             't': 'excess return t'})


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



def monitor_process():
    pass


