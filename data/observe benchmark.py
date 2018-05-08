# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-01  16:56
# NAME:assetPricing2-observe 11 benchmark.py
from data.dataApi import Database
from data.dataTools import load_data

DATA=Database(sample_control=False)

data=DATA.data

benchs=['capmM', 'ff3M', 'ffcM', 'ff5M', 'hxz4M']

for bench in benchs:
    d=DATA.by_factor(bench)

    fig=d.cumsum().plot().get_figure()
    fig.savefig(r'e:\a\{}.png'.format(bench))


bench='capmM'
d=DATA.by_factor(bench)

d.describe()

d.head()



