#-*-coding: utf-8 -*-
#author:tyhj
#fama_macbeth_Vincent_Grégoire.py 2017.10.25 21:42
import pandas as pd
import statsmodels.stats.sandwich_covariance as sw
import statsmodels.formula.api as sm
import numpy as np
from urllib import urlopen
url=r'http://www.kellogg.northwestern.edu/faculty/petersen/htm/papers/se/test_data.txt'
df=pd.read_table(urlopen(url),names=['firmid','year','x','y'],
                delim_whitespace=True)
df.head()


def fama_macbeth(formula, time_label, df, lags=3):
    res = df.groupby(time_label).apply(lambda x: sm.ols(
        formula, data=x).fit())
    l = [x.params for x in res]
    p = pd.DataFrame(l)

    means = {}
    params_labels = res.iloc[0].params.index

    # The ':' character used by patsy doesn't play well with pandas
    # column indicators.
    p.columns = [x.replace(':', '_INTER_') for x in p.columns]

    for x in p.columns:
        if lags is 0:
            means[x.replace('_INTER_', ':')] = sm.ols(formula=x + ' ~ 1',
                                                      data=p[[x]]).fit(use_t=True)
        else:
            means[x.replace('_INTER_', ':')] = sm.ols(formula=x + ' ~ 1',
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

    return result

fama_macbeth('y ~ x','year',df)

#forgo the Newey-West correction by passing lags=0
fama_macbeth('y ~ x','year',df,lags=0)














