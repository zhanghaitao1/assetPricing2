# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-03  10:23
# NAME:assetPricing2-analyse_factors.py

from config import PROJECT_PATH
import os
import pandas as pd
from dataset import DATA

from old.main import OneFactor


class SIZE(OneFactor):
    path = os.path.join(PROJECT_PATH, 'size')

    def __init__(self):
        super().__init__('size',self.path)

    def get_percent_ratio(self):
        '''Fig 9.1  page 152'''
        def _get_ratio(s):
            s=s.dropna()
            total=s.shape[0]
            ratios = [0.01, 0.05, 0.10, 0.25]
            num=[int(r*total) for r in ratios]
            return pd.Series([s.nlargest(n).sum() / s.sum() for n in num],
                             index=ratios)

        df=DATA.by_indicators('mktCap')
        d=df.groupby('t')['mktCap'].apply(_get_ratio)
        fig=d.unstack().plot().get_figure()
        fig.savefig(os.path.join(self.path,'percent of market value.png'))

    def __call__(self):
        super().run()
        self.get_percent_ratio()


class VALUE(OneFactor):
    path=os.path.join(PROJECT_PATH,'value')

    def __init__(self):
        super().__init__('value',self.path)

    def correlation(self,indicators=('bm','lgbm','D_12M','size')):
        super().correlation(indicators)


class MOM(OneFactor):
    path=os.path.join(PROJECT_PATH,'mom')

    def correlation(self,indicators=DATA.info['momentum']+['D_12M','size','bm']):
        super().correlation(indicators)

    pass




if __name__=='__main__':
    size=SIZE()
    size()

