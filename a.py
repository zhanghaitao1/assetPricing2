# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-08  10:57
# NAME:assetPricing2-a.py

import pandas as pd
import numpy as np

index=pd.date_range('1/1/2000',periods=8,freq='T')

#TODO:how about np.nan
df=pd.DataFrame({'s':[0.0,None,2.0,3.0,np.nan,np.nan,6.0,7.0]},index=index)



df.asfreq(freq='30S')

df.asfreq(freq='30S',fill_value=9.0)
df.asfreq(freq='30S',method='bfill')

df.resample('2T').last()
df.resample('2T').agg(lambda x:x[-1])

df.resample('2T').agg(lambda x:x[0])
df.resample('2T').mean()

df.asfreq(freq='2T')
