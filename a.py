# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-08  10:57
# NAME:assetPricing2-a.py


from data.dataApi import Database
from data.outlier import detect_outliers

DATA=Database()

detect_outliers(DATA.data,'data')





