# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-08  17:49
# NAME:assetPricing2-analyse.py

import pandas as pd
from cakici.data import PATH
import os

factors=pd.read_csv(os.path.join(PATH,'factors.csv'),index_col=[0,1],parse_dates=True)

variables=['size','bkmt','price','sd','see']
df=factors[variables]
by_year=df.resample('Y',level='t').agg(lambda x:x.mean())




