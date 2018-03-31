# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-22  15:09
# NAME:assetPricing2-main.py

from dataset import dataset
from zht.utils import assetPricing
from zht.utils.assetPricing import summary_statistics, cal_breakPoints, count_groups
import os
import pandas as pd
import numpy as np

class OneFactor:
    q=10

    def __init__(self, factor,path):
        self.factor=factor
        self.path=path
        self.indicators=dataset.info[factor]
        self.df=dataset.by_factor(factor)
        self.groupnames=[self.factor+str(i) for i in range(1,self.q+1)]

    def summary(self):
        series=[]
        for indicator in self.indicators:
            s=summary_statistics(self.df[indicator].unstack())
            series.append(s.mean())
        pd.concat(series,keys=self.indicators,axis=1).to_csv(os.path.join(self.path,'summary.csv'))

    def correlation(self,indicators=None):
        if not indicators:
            indicators=self.indicators

        comb=dataset.get_by_indicators(indicators)
        def _spearman(df):
            df=df.dropna()
            if df.shape[0]>10:#TODO:thresh to choose
                return assetPricing.corr(df,'spearman',winsorize=False)

        def _pearson(df):
            df=df.dropna()
            if df.shape[0]>10:
                return assetPricing.corr(df,'pearson',winsorize=True)

        corrs=comb.groupby('t').apply(_spearman)
        corrp=comb.groupby('t').apply(_pearson)

        corrsAvg=corrs.groupby(level=1).mean().reindex(index=indicators, columns=indicators)
        corrpAvg=corrp.groupby(level=1).mean().reindex(index=indicators, columns=indicators)

        corr1 = np.tril(corrpAvg.values, k=-1)
        corr2 = np.triu(corrsAvg.values, k=1)

        corr = pd.DataFrame(corr1 + corr2, index=corrpAvg.index, columns=corrpAvg.columns)
        np.fill_diagonal(corr.values, np.NaN)
        corr.to_csv(os.path.join(self.path, 'corr.csv'))

    def persistence(self):
        #TODO: Table II of Asness, Clifford S., Andrea Frazzini, and Lasse Heje Pedersen. “Quality Minus Junk.” SSRN Scholarly Paper. Rochester, NY: Social Science Research Network, June 5, 2017. https://papers.ssrn.com/abstract=2312432.

        perdf=pd.DataFrame()
        for indicator in self.indicators:
            per=assetPricing.persistence(self.df[indicator].unstack(),
                                         offsets=[1, 3, 6, 12, 24, 36, 48, 60, 120])
            perdf[indicator]=per
        perdf.to_csv(os.path.join(self.path,'persistence.csv'))

    def breakPoints_and_countGroups(self):
        for indicator in self.indicators:
            d=self.df[indicator].unstack()
            d=d.dropna(axis=0,how='all')#there is no samples for some months due to festival
            bps=cal_breakPoints(d,self.q)
            bps.to_csv(os.path.join(self.path,'breakPoints_%s.csv'%indicator))
            count=count_groups(d,self.q)
            count.to_csv(os.path.join(self.path,'count_%s.csv'%indicator))
        #TODO: put them in one csv (stacked)

        # TODO:In fact,the count is not exactly the number of stocks to calculate the weighted return
        # TODO:as some stocks will be deleted due to the missing of weights.

    def _get_port_data(self,indicator):
        groupid=dataset.get_by_indicators([indicator])
        groupid['g']=groupid.groupby('t',group_keys=False).apply(
            lambda df:pd.qcut(df[indicator],self.q,
                              labels=[indicator+str(i) for i in range(1,self.q+1)])
        )
        return groupid

    def portfolio_characteristics(self,sortedIndicator,otherIndicators):
        '''
        as table 12.3 panel A
        :param sortedIndicator:
        :param otherIndicators:
        :return:
        '''
        groupid=self._get_port_data(sortedIndicator)
        comb=dataset.get_by_indicators(otherIndicators)
        comb=pd.concat([groupid,comb],axis=1)
        characteristics_avg=comb.groupby(['t','g']).mean().groupby('g').mean()
        characteristics_avg.to_csv(os.path.join(self.path,'portfolio characteristics.csv'))
