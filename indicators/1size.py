# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-21  14:14
# NAME:assetPricing2-2 size.py

from dout import *
from config import *
import numpy as np


def cal_sizes():
    '''
    compute variety of sizes

    :return:
    '''
    mktCap=read_df('capM',freq='M')
    mktCap[mktCap<=0]=np.nan #TODO:filter out invalid data
    size=np.log(mktCap)
    size.to_csv(os.path.join(DATA_PATH,'size.csv'))

    # junes=[m for m in mktCap.index.tolist() if m.split('-')[1]=='06']
    #-------
    junes = [m for m in mktCap.index.tolist() if m.month == 6]
    newIndex=pd.date_range(start=junes[0],end=mktCap.index[-1],freq='M')
    junesDf = mktCap.loc[junes]
    mktCap_ff=junesDf.reindex(index=newIndex)
    mktCap_ff=mktCap_ff.ffill(limit=11)#limit=11 is required,or it will fill all NaNs forward.
    mktCap_ff.to_csv(os.path.join(DATA_PATH,'mktCap_ff.csv'))

    size_ff=np.log(mktCap_ff)
    size_ff.to_csv(os.path.join(DATA_PATH,'size_ff.csv'))


if __name__=='__main__':
    cal_sizes()





