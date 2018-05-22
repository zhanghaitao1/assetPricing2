# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-24  17:30
# NAME:assetPricing2-din_filter_outliers.py

import pandas as pd
import numpy as np
import os

from config import PKL_UNFILTERED_PATH
from data.dataApi import Database
from data.dataTools import read_unfiltered, save_to_filtered, load_data
from data.outlier import detect_outliers, delete_outliers

#TODO: Do not apply_condition in this module

def handle_outliers(tbname):
    x=read_unfiltered(tbname)
    detect_outliers(x,tbname)
    x1=delete_outliers(x,'mad',6)
    detect_outliers(x1,'filtered_'+tbname)
    save_to_filtered(x1,tbname)

def filter_all():
    tbnames=[fn[:-4] for fn in os.listdir(PKL_UNFILTERED_PATH)]
    for tbname in tbnames:
        try:
            handle_outliers(tbname)
            print(tbname)
        except:
            print('{} is skipped'.format(tbname))






