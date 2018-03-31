# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-21  14:14
# NAME:assetPricing2-size.py

from dout import *
from config import *
import numpy as np


def cal_sizes():
    '''
    compute variety of sizes

    :return:
    '''
    mktCap=read_df('mktCap',freq='M')
    mktCap[mktCap<=0]=np.nan
    size=np.log(mktCap)
    size.to_csv(os.path.join(DATA_PATH,'size.csv'))

    # junes=[m for m in mktCap.index.tolist() if m.split('-')[1]=='06']
    mths=[m for m in mktCap.index.tolist() if m.month==6]+[mktCap.index[-1]]
    junesDf=mktCap.loc[mths]
    mktCap_ff=junesDf.resample('M').ffill()
    mktCap_ff.to_csv(os.path.join(DATA_PATH,'mktCap_ff.csv'))

    size_ff=np.log(mktCap_ff)
    size_ff.to_csv(os.path.join(DATA_PATH,'size_ff.csv'))


if __name__=='__main__':
    cal_sizes()





