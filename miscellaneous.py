# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-22  14:55
# NAME:assetPricing2-miscellaneous.py


from dout import *
from zht.utils.mathu import get_inter_index


def compare_ff5_ff3_ffc():
    ff3=read_df('ff3M','M')
    ff5=read_df('ff5M','M')
    ffc=read_df('ffcM','M')

    [ff3,ff5,ffc]=get_inter_index([ff3,ff5,ffc])

    ff3.cumsum().plot()
    ff5.cumsum().plot()
    ffc.cumsum().plot()

compare_ff5_ff3_ffc()

