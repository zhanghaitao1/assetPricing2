# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-22  15:52
# NAME:assetPricing2-reversal.py

from data.dataTools import load_data, save_to_filter


def get_rev():
    stockRetM=load_data('stockRetM')
    rev=stockRetM*100
    rev=rev.stack().to_frame()
    rev.columns=['reversal']
    rev.index.names=['t','sid']
    save_to_filter(rev,'reversal')


if __name__=='__main__':
    get_rev()

