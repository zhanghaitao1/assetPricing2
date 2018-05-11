# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-21  17:15
# NAME:assetPricing2-dout.py

import pandas as pd
import os

from config import CSV_UNFILTERED_PATH, PKL_UNFILTERED_PATH
from data.dataTools import read_unfiltered




def read_data(tbname):
    x=read_unfiltered(tbname)


