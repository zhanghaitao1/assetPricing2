# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-21  14:18
# NAME:assetPricing2-3value.py


from config import *
from dout import read_df
from zht.data.gta.api import read_gta
from zht.data.wind.api import read_wind
from zht.utils.mathu import get_inter_frame
import numpy as np
import pandas as pd


#compare the bps of wind with bv of gta
def compare_wind_gta_bps():
    '''
    the result is different a lot!!!

    :return:
    '''
    bps_wind=read_df('bps_wind','M')
    bps_gta=read_df ('bps','M')
    bps_wind.columns=[str(int(col[:-3])) for col in bps_wind.columns]
    bps_wind=bps_wind.sort_index(axis=1)
    bps_gta=bps_gta.sort_index(axis=1)
    bps_wind,bps_gta=get_inter_frame([bps_wind,bps_gta])


def get_bm():
    '''
    this function can be bookmarked as a snippet of how to manipulate date index
    in Pandas

    A little different with the book,here we use be and me for one share,
    but the data in the book is for all floating shares.However,it doesn't
    affect the bm.

    :return:
    '''
    be=read_df('bps','M')
    be=be[be.index.month==12]
    me=read_df('stockCloseY','M')
    be,me=get_inter_frame([be,me])
    bm=be/me
    bm[bm<=0]=np.nan #delete those sample with bm<0
    bm=bm.shift(1,freq='6M')

    newIndex=pd.date_range(bm.index[0],bm.index[-1],freq='M')
    bm=bm.reindex(index=newIndex)
    bm=bm.fillna(method='ffill',limit=11)
    bm.to_csv(os.path.join(DATA_PATH,'bm.csv'))

    logbm=np.log(bm)
    logbm.to_csv(os.path.join(DATA_PATH,'logbm.csv'))


if __name__=='__main__':
    get_bm()

