# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-08  10:57
# NAME:assetPricing2-a.py

import pandas as pd
from pandas.tseries.offsets import MonthEnd

date=pd.date_range('2018-02-01','2018-04-09',freq='D')
s=pd.Series(range(len(date)),index=date)
s=s[:3].append(s[30:32]).append(s[-3:])
s.asfreq('M')
s.resample('M').agg(lambda x:x[-1])


s.groupby(lambda x:x+MonthEnd(0)).agg(lambda x:x[-1])
s.groupby(lambda x:x+MonthEnd(0)).agg(lambda x:x[0])

