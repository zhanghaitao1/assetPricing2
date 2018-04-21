# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-21  19:58
# NAME:assetPricing2-base.py


class MyError(Exception):
    '''The name of index is wrong'''
    def __init__(self,msg):
        super().__init__(msg)
