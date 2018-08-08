# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-22  16:00
# NAME:assetPricing2-7 skewness.py
import os



import pandas as pd

path=r'D:\zht\database\quantDb\researchTopics\assetPricing2_new\playingfield\apply_condition_result\skewness'
fns=os.listdir(path)
fns=[fn for fn in fns if fn.endswith('.csv')]

# excelPath=r'e:\a\test.xlsx'
excelWriter=pd.ExcelWriter(r'e:\a\test.xlsx')
for fn in fns:
    df=pd.read_csv(os.path.join(path,fn))
    df.to_excel(excelWriter,fn[-10:])

