# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-07-24  10:39
# NAME:assetPricing2-combine_data.py
from data.dataTools import read_filtered

size=read_filtered('size')['size']
price=read_filtered('stockCloseM').stack()
beta=read_filtered('beta_sw_dm')
sd=read_filtered('sd')
see=read_filtered('see')


