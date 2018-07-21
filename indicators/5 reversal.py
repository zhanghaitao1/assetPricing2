# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-22  15:52
# NAME:assetPricing2-5 reversal.py

from data.dataTools import save, read_unfiltered


def get_rev():
    stockRetM=read_unfiltered('stockRetM')
    # stockRetM=load_data('stockRetM')
    rev=stockRetM*100
    save(rev,'reversal')

if __name__=='__main__':
    get_rev()

