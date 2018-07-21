# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-20  19:33
# NAME:assetPricing2-config.py


import os
DATA_SRC=r'D:\zht\database\quantDb\sourceData\gta\data\csv'

PROJECT_PATH=r'D:\zht\database\quantDb\researchTopics\assetPricing3'
DATA_PATH=os.path.join(PROJECT_PATH, 'data')

TMP_PATH=os.path.join(PROJECT_PATH,'tmp')
WINSORIZE_LIMITS=(0.01,0.01)

CSV_UNFILTERED_PATH=os.path.join(DATA_PATH, 'csv_unfiltered')
PKL_UNFILTERED_PATH=os.path.join(DATA_PATH, 'pkl_unfiltered')

OUTLIER_PATH=os.path.join(DATA_PATH,'outlier')
CSV_FILTERED_PATH=os.path.join(DATA_PATH,'csv_filtered')
PKL_FILTERED_PATH=os.path.join(DATA_PATH, 'pkl_filtered')




