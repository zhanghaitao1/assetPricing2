# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-21  17:13
# NAME:assetPricing2-2 size.py

import numpy as np
import pandas as pd

from data.dataTools import load_data, save_to_filtered, save


def cal_sizes():
    mktCap=load_data('capM')
    mktCap[mktCap<=0]=np.nan
    size=np.log(mktCap)

    junes = [m for m in mktCap.index.tolist() if m.month == 6]
    newIndex = pd.date_range(start=junes[0], end=mktCap.index[-1], freq='M')
    junesDf = mktCap.loc[junes]
    mktCap_ff = junesDf.reindex(index=newIndex)
    mktCap_ff = mktCap_ff.ffill(limit=11)  # limit=11 is required,or it will fill all NaNs forward.

    size_ff = np.log(mktCap_ff)

    size=size.stack()
    size.name='size'
    mktCap_ff=mktCap_ff.stack()
    mktCap_ff.name='mktCap_ff'
    size_ff=size_ff.stack()
    size_ff.name='size_ff'

    mktCap=mktCap.stack()
    mktCap.name='mktCap'
    # combine
    x=pd.concat([mktCap,mktCap_ff,size,size_ff],axis=1)
    x.index.names=['t','sid']
    x.columns.name='type'

    save(x,'size')



if __name__ == '__main__':
    cal_sizes()

