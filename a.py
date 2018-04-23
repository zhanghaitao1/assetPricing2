# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-08  10:57
# NAME:assetPricing2-a.py


import inspect


def func(x):
    return x*x

args=range(10)

str_func=inspect.getsource(func)
str_args=inspect.getsource(args)

str_main=open('multi_base.py').read()

with open(r'e:\a\test.py','w') as f:
    f.write('\n'.join([str_func,str_args,str_main]))



