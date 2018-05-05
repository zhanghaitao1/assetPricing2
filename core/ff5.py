# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-05  23:45
# NAME:assetPricing2-ff5.py

'''
references:
    1. Fama, E.F., and French, K.R. (1993). Common risk factors in the returns
    on stocks and bonds. Journal of Financial Economics 33, 3–56.
    2. Fama, E.F., and French, K.R. (2015). A five-factor asset pricing model.
    Journal of Financial Economics 116, 1–22.

'''




# playing field
from core.timeSeriesRegression import data_for_bivariate
from data.dataTools import load_data
from tool import my_average


v1='size__size'
v2='value__logbm'

comb=data_for_bivariate(v1,v2,5,5,independent=True)

portfolios=comb.groupby(['t','g1','g2']).apply(
    lambda df:my_average(df,'stockEretM',wname='weight'))


portfolios.unstack(level=['g1','g2'])


ff3=load_data('ff3M')


#GRS



