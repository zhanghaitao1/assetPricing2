# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-24  17:30
# NAME:assetPricing2-din_filter_outliers.py

import pandas as pd
import numpy as np

from data.dataTools import read_unfiltered, save_to_filtered, load_data
from data.outlier import detect_outliers
from data.sampleControl import apply_condition, start_end


#TODO: apply condition before detecting outliers or detect outliers before applying conditions?
#TODO:

def filter_stockRetD():
    raw=read_unfiltered('stockRetD')
    detect_outliers(raw,'stockRetD')
    x=apply_condition(raw)
    x[abs(x)>0.11]=np.nan

    save_to_filtered(x, 'stockRetD')

def filter_stockCloseD():
    name='stockCloseD'
    raw=read_unfiltered(name)
    x=apply_condition(raw)
    save_to_filtered(x, name)

def filter_mktRetD():
    raw=read_unfiltered('mktRetD')
    x=start_end(raw)
    save_to_filtered(x, 'mktRetD')

def filter_rfD():
    raw=read_unfiltered('rfD')
    x=start_end(raw)
    save_to_filtered(x, 'rfD')

def filter_rfM():
    raw = read_unfiltered('rfM')
    x = start_end(raw)
    save_to_filtered(x, 'rfM')

def filter_stockEretD():
    stockRetD=load_data('stockRetD')
    rfD=load_data('rfD')
    stockEretD=stockRetD.sub(rfD,axis=0)
    save_to_filtered(stockEretD, 'stockEretD')

def filter_stockRetM():
    raw=read_unfiltered('stockRetM')
    x=apply_condition(raw)
    x[abs(x)>1.0]=np.nan

    save_to_filtered(x, 'stockRetM')

def filter_stockCloseM():
    raw=read_unfiltered('stockCloseM')
    x=apply_condition(raw)
    save_to_filtered(x, 'stockCloseM')

def filter_stockEretM():
    stockRetM=load_data('stockRetM')
    rfM=load_data('rfM')
    stockEretM=stockRetM.sub(rfM,axis=0)
    save_to_filtered(stockEretM, 'stockEretM')

def filter_mktRetM():
    raw=read_unfiltered('mktRetM')
    x=start_end(raw)
    save_to_filtered(x, 'mktRetM')

def filter_capM():
    raw=read_unfiltered('capM')
    x=apply_condition(raw)
    save_to_filtered(x, 'capM')

def filter_bps():
    raw=read_unfiltered('bps')
    x=apply_condition(raw)
    x[abs(x)>100]=np.nan #TODO:

    save_to_filtered(x, 'bps')

def filter_bps_wind():
    raw=read_unfiltered('bps_wind')
    x=start_end(raw)
    x=x.where(x<100.0)
    save_to_filtered(x, 'bps_wind')

def filter_stockCloseY():
    raw=read_unfiltered('stockCloseY')
    x=apply_condition(raw)
    detect_outliers(x,'stockCloseY1')
    save_to_filtered(x, 'stockCloseY')

def filter_ff3M_resset():
    raw=read_unfiltered('ff3M_resset')
    x=start_end(raw)

    save_to_filtered(x, 'ff3M_resset')

def filter_ff3M():
    raw=read_unfiltered('ff3M')
    x=start_end(raw)
    for col,s in x.iteritems():
        detect_outliers(s,col)

    save_to_filtered(x, 'ff3M')

def filter_ffcM():
    raw=read_unfiltered('ffcM')
    x=start_end(raw)
    for col,s in x.iteritems():
        detect_outliers(s,'ffcM_'+col)

    save_to_filtered(x, 'ffcM')

def filter_ff5M():
    raw=read_unfiltered('ff5M')
    x=start_end(raw)

    for col,s in x.iteritems():
        detect_outliers(s,'ff5M_'+col)

    save_to_filtered(x, 'ff5M')

def filter_hxz4M():
    raw = read_unfiltered('hxz4M')
    x = start_end(raw)

    for col, s in x.iteritems():
        detect_outliers(s, 'hxz4M_' + col)

    save_to_filtered(x, 'hxz4M')

def filter_ff3D():
    raw = read_unfiltered('ff3D')
    x = start_end(raw)

    for col, s in x.iteritems():
        detect_outliers(s, 'ff3D_' + col)
    #TODO:noisy
    save_to_filtered(x, 'ff3D')

def filter_fpM():
    raw=load_data('ff3M')['rp']
    raw.name='rpM'
    save_to_filtered(raw, 'rpM')

def filter_rpD():
    raw=load_data('ff3D')['rp']
    raw.name='rpD'
    save_to_filtered(raw, 'rpD')
