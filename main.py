# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-22  15:09
# NAME:assetPricing2-main.py

from dataset import DATA
from tool import adjust_with_riskModel, grouping
from zht.utils import assetPricing
from zht.utils.assetPricing import summary_statistics, cal_breakPoints, count_groups, famaMacBeth
import os
import pandas as pd
import numpy as np
from zht.utils.mathu import get_outer_frame


def asign_port_id(s, q, labels, thresh=None):
    '''
    this function will first dropna and then asign porfolio id.

    :param s: Series
    :param q:
    :param labels:
    :param thresh:
    :return:
    '''
    ns = s.dropna()
    if thresh is None:
        thresh = q * 10  # TODO: thresh self.q*10？

    if ns.shape[0] > thresh:
        result = pd.qcut(ns, q, labels)
        return result
    else:
        return pd.Series(index=ns.index)


class OneFactor:
    q=10

    def __init__(self, factor,path):
        self.factor=factor
        self.path=path
        self.indicators=DATA.info[factor]
        self.df=DATA.data[self.indicators]#
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

        comb=DATA.by_indicators(indicators)
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
            d=d.dropna(axis=0,how='all',thresh=self.q*10)#there is no samples for some months due to festival
            bps=cal_breakPoints(d,self.q)
            bps.to_csv(os.path.join(self.path,'breakPoints_%s.csv'%indicator))
            count=count_groups(d,self.q)
            count.to_csv(os.path.join(self.path,'count_%s.csv'%indicator))
        #TODO: put them in one csv (stacked)

        # TODO:In fact,the count is not exactly the number of stocks to calculate the weighted return
        # TODO:as some stocks will be deleted due to the missing of weights.

    def _get_port_data(self,indicator):
        groupid=DATA.by_indicators([indicator])
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
        comb=DATA.by_indicators(otherIndicators)
        comb=pd.concat([groupid,comb],axis=1)
        characteristics_avg=comb.groupby(['t','g']).mean().groupby('g').mean()
        characteristics_avg.to_csv(os.path.join(self.path,'portfolio characteristics.csv'))

    #TODO: upgrade this function
    def _get_panel_stk_avg(self, comb, indicator, gcol):
        panel_stk_eavg=comb.groupby(['t',gcol])['eretM'].mean()
        if self.factor=='size':
            panel_stk_wavg=comb.groupby(['t',gcol]).apply(
                lambda df:np.average(df['eretM'],weights=df[indicator])
            )
        else:
            panel_stk_wavg = comb.groupby(['t', gcol]).apply(
                lambda df: np.average(df['eretM'], weights=df['capM'])
                # TODO:how about the nan
            )
        return panel_stk_eavg,panel_stk_wavg

    #TODO: something wrong!!!!!
    def portfolio_analysis(self):
        #TODO: add a parameter to declare what risk models will be used. [ff3,capm,ff5]

        all_indicators = list(set(self.indicators + ['capM', 'eretM', 'mktRetM']))
        comb = DATA.by_indicators(all_indicators)

        result_eavg=[]
        result_wavg=[]
        for indicator in self.indicators:
            gcol='g_%s'%indicator
            # comb[gcol]=comb.groupby('t').apply(
            #     lambda df:grouping(df[indicator].reset_index(level='t'),self.q,labels=self.groupnames))
            comb[gcol]=comb.groupby('t',group_keys=False).apply(
                lambda df:asign_port_id(df[indicator],self.q,self.groupnames))

            panel_stk_eavg,panel_stk_wavg=self._get_panel_stk_avg(comb, indicator, gcol)
            for panel_stk in [panel_stk_eavg,panel_stk_wavg]:
                panel=panel_stk.unstack(level=[gcol])
                panel.columns=panel.columns.astype(str)
                panel['_'.join([self.groupnames[-1],self.groupnames[0]])]=panel[self.groupnames[-1]]-panel[self.groupnames[0]]
                panel['avg']=panel.mean(axis=1)
                #TODO: use the risk models declared above

                a_data = comb.groupby(['t', gcol])[indicator].mean()
                a_data = a_data.unstack()
                a_data.columns = a_data.columns.astype(str)
                a_data.index = a_data.index.astype(str)
                a_data['_'.join([self.groupnames[-1], self.groupnames[0]])] = a_data[self.groupnames[-1]] - a_data[
                    self.groupnames[0]]
                a_data['avg']=a_data.mean(axis=1)
                a = a_data.mean()
                a.name='avg'
                a=a.to_frame().T
                b=adjust_with_riskModel(panel)
                c=adjust_with_riskModel(panel,riskmodel='capm')

                if panel_stk is panel_stk_eavg:
                    result_eavg.append(pd.concat([a,b,c],axis=0))
                else:
                    result_wavg.append(pd.concat([a,b,c],axis=0))
        table_e=pd.concat(result_eavg,axis=0,keys=self.indicators)
        table_w=pd.concat(result_wavg,axis=0,keys=self.indicators)
        table_e.to_csv(os.path.join(self.path,'univariate portfolio analysis-equal weighted.csv'))
        table_w.to_csv(os.path.join(self.path,'univariate portfolio analysis-value weighted.csv'))

    def fm(self):
        comb=DATA.by_indicators(self.indicators+['eretM'])
        data=[]
        for indicator in self.indicators:
            subdf=comb[[indicator,'eretM']]
            subdf=subdf.dropna()
            subdf.columns=['y','x']
            subdf=subdf.reset_index()
            formula='y ~ x'
            r,adj_r2,n=famaMacBeth(formula,'t',subdf,lags=5)
            data.append([r.loc['x', 'coef'], r.loc['x', 'tvalue'],
                         r.loc['Intercept', 'coef'], r.loc['Intercept', 'tvalue'],
                         adj_r2, n])

        result = pd.DataFrame(data, index=self.indicators,
                              columns=['slope', 't', 'Intercept', 'Intercept_t', 'adj_r2', 'n']).T
        result.to_csv(os.path.join(self.path, 'fama macbeth regression analysis.csv'))

    def run(self):
        self.summary()
        self.correlation()
        self.persistence()
        self.breakPoints_and_countGroups()
        # self.portfolio_characteristics()
        self.portfolio_analysis()
        self.fm()

    def __call__(self):
        pass

class Bivariate:
    q=5
    def __init__(self,indicator1,indicator2,proj_path):
        self.indicator1=indicator1
        self.indicator2=indicator2
        self.path=proj_path

    def _get_independent_data(self):
        # TODO: add the method of ratios such as [0.3,0.7]
        # sometimes the self.indicators and ['capM','eretM'] may share some elements
        comb=DATA.by_indicators([self.indicator1,self.indicator2,'capM','eretM'])
        comb=comb.dropna()
        comb['g1']=comb.groupby('t',group_keys=False).apply(
            lambda df:asign_port_id(df[self.indicator1],self.q,
                                    [self.indicator1 + str(i) for i in range(1, self.q + 1)]))

        comb['g2']=comb.groupby('t',group_keys=False).apply(
            lambda df:asign_port_id(df[self.indicator2],self.q,
                                    [self.indicator2 + str(i) for i in range(1,self.q + 1)]))

        # comb['g1']=comb.groupby('t',group_keys=False).apply(
        #     lambda df:pd.qcut(df[self.indicator1],self.q,
        #                       labels=[self.indicator1+str(i) for i in range(1,self.q+1)])
        # )
        #
        # comb['g2']=comb.groupby('t',group_keys=False).apply(
        #     lambda df:pd.qcut(df[self.indicator2],self.q,
        #                       labels=[self.indicator2+str(i) for i in range(1,self.q+1)])
        # )

        return comb

    def _get_dependent_data(self,indicators):
        '''

        :param indicators:list with two elements,the first is the controlling variable
        :return:
        '''

        # sometimes the self.indicators and ['mktCap','eret'] may share some elements
        comb=DATA.by_indicators(indicators+['capM','eretM'])
        comb=comb.dropna()
        comb['g1']=comb.groupby('t',group_keys=False).apply(
            lambda df:asign_port_id(df[indicators[0]],self.q,
                                    [indicators[0] + str(i) for i in range(1,self.q + 1)]))

        comb['g2']=comb.groupby(['t','g1'],group_keys=False).apply(
            lambda df:asign_port_id(df[indicators[1]],self.q,
                                    [indicators[1] + str(i) for i in range(1,self.q + 1)]))

        return comb

    def _get_eret(self,comb):
        group_eavg_ts = comb.groupby(['g1', 'g2', 't'])['eretM'].mean()
        group_wavg_ts = comb.groupby(['g1', 'g2', 't']).apply(
            lambda df: np.average(df['eretM'], weights=df['capM']))
        return group_eavg_ts,group_wavg_ts

    def _dependent_portfolio_analysis(self, group_ts, controlGroup='g1', targetGroup='g2'):
        # Table 9.6
        controlIndicator = group_ts.index.get_level_values(controlGroup)[0][:-1]
        targetName = group_ts.index.get_level_values(targetGroup)[0][:-1]

        # A
        a_data = group_ts.groupby(['t', controlGroup, targetGroup]).mean().unstack(level=[controlGroup])
        a_data.columns = a_data.columns.astype(str)

        # A1
        a1_data = group_ts.groupby(['t', controlGroup, targetGroup]).mean().groupby(['t', targetGroup]).mean()
        a_data[controlIndicator + ' avg'] = a1_data
        _a = a_data.groupby(targetGroup).mean()

        def _get_spread(df):
            time = df.index.get_level_values('t')[0]
            return df.loc[(time, targetName + str(self.q))] - df.loc[(time, targetName + '1')]

        # B
        b_data = a_data.groupby('t').apply(_get_spread)

        _b1=adjust_with_riskModel(b_data)
        _b2=adjust_with_riskModel(b_data,'capm')

        _b1.index = [targetName + str(self.q) + '-1', targetName + str(self.q) + '-1 t']
        _b2.index = [targetName + str(self.q) + '-1 capm alpha', targetName + str(self.q) + '-1 capm alpha t']

        _a.index = _a.index.astype(str)
        _a.columns = _a.columns.astype(str)

        return pd.concat([_a, _b1, _b2], axis=0)

    def _dependent_portfolio_analysis_twin(self, group_ts, controlGroup='g2', targetGroup='g1'):
        # table 10.5 panel B
        targetIndicator = group_ts.index.get_level_values(targetGroup)[0][:-1]  # targetGroup
        # controlIndicator = group_ts.index.get_level_values(controlGroup)[0][:-1]  # controlGroup

        a1_data = group_ts.groupby(['t', targetGroup, controlGroup]).mean().groupby(['t', targetGroup]).mean()

        stk = a1_data.unstack()
        stk.index = stk.index.astype(str)
        stk.columns = stk.columns.astype(str)
        stk[targetIndicator + str(self.q) + '-1'] = stk[targetIndicator + str(self.q)] - stk[targetIndicator + '1']

        _a=adjust_with_riskModel(stk)
        _b=adjust_with_riskModel(stk,'capm')

        table = pd.concat([_a, _b], axis=0)
        return table

    def _independent_portfolio_analysis(self, group_ts):
        # table 9.8
        table1 = self._dependent_portfolio_analysis(group_ts, controlGroup='g1', targetGroup='g2')
        table2 = self._dependent_portfolio_analysis(group_ts, controlGroup='g2', targetGroup='g1').T
        table1, table2 = get_outer_frame([table1, table2])
        table = table1.fillna(table2)
        return table

    def independent_portfolio_analysis(self):
        comb = self._get_independent_data()
        group_eavg_ts, group_wavg_ts = self._get_eret(comb)

        table_eavg = self._independent_portfolio_analysis(group_eavg_ts)
        table_wavg = self._independent_portfolio_analysis(group_wavg_ts)
        table_eavg.to_csv(os.path.join(self.path,
                                       'bivariate independent-sort portfolio analysis_equal weighted_%s_%s.csv' % (
                                       self.indicator1, self.indicator2)))
        table_wavg.to_csv(os.path.join(self.path,
                                       'bivariate independent-sort portfolio analysis_value weighted_%s_%s.csv' % (
                                       self.indicator1, self.indicator2)))

    def dependent_portfolio_analysis(self):
        def _f(indicators):
            comb = self._get_dependent_data(indicators)
            group_eavg_ts, group_wavg_ts = self._get_eret(comb)

            table_eavg = self._dependent_portfolio_analysis(group_eavg_ts)
            table_wavg = self._dependent_portfolio_analysis(group_wavg_ts)
            table_eavg.to_csv(os.path.join(self.path,
                                           'bivariate dependent-sort portfolio analysis_equal weighted_%s_%s.csv' % (
                                           indicators[0], indicators[1])))
            table_wavg.to_csv(os.path.join(self.path,
                                           'bivariate dependent-sort portfolio analysis_value weighted_%s_%s.csv' % (
                                           indicators[0], indicators[1])))

        _f([self.indicator1,self.indicator2])
        _f([self.indicator2,self.indicator1])

    def dependent_portfolio_analysis_twin(self):
        def _f(indicators):
            comb = self._get_dependent_data(indicators)
            group_eavg_ts, group_wavg_ts = self._get_eret(comb)

            table_eavg = self._dependent_portfolio_analysis_twin(group_eavg_ts)
            table_wavg = self._dependent_portfolio_analysis_twin(group_wavg_ts)

            table_eavg.to_csv(os.path.join(self.path,
                                           'bivariate dependent-sort portfolio analysis_twin_equal weighted_%s_%s.csv' % (
                                               indicators[0], indicators[1])))
            table_wavg.to_csv(os.path.join(self.path,
                                           'bivariate dependent-sort portfolio analysis_twin_weighted_%s_%s.csv' % (
                                               indicators[0], indicators[1])))

        _f([self.indicator1, self.indicator2])
        _f([self.indicator2, self.indicator1])

    def _fm(self, ll_indeVars):
        '''
        :param ll_indeVars: list of list,the inside list contains all
            the indepedent variables to construct a regress equation

        :return:
        '''
        indicators = list(set(var for l_indeVars in ll_indeVars for var in l_indeVars)) + ['eretM']
        comb = DATA.by_indicators(indicators)
        comb = comb.reset_index()

        stks = []
        for l_indeVars in ll_indeVars:
            '''
            replace the olde name with new name,since patsy do not support name starts with number 

            '''
            newname = ['name' + str(i) for i in range(1, len(l_indeVars) + 1)]
            df = comb[l_indeVars + ['t', 'eretM']].dropna()
            df.columns = newname + ['t', 'eretM']
            formula = 'eretM ~ ' + ' + '.join(newname)
            # TODO:lags?
            r, adj_r2, n = famaMacBeth(formula, 't', df, lags=5)
            r = r.rename(index=dict(zip(newname, l_indeVars)))
            stk = r[['coef', 'tvalue']].stack()
            stk.index = stk.index.map('{0[0]} {0[1]}'.format)
            stk['adj_r2'] = adj_r2
            stk['n'] = n
            stks.append(stk)

        table = pd.concat(stks, axis=1, keys=range(1, len(ll_indeVars) + 1))

        newIndex = [var + ' ' + suffix for var in indicators for suffix in ['coef', 'tvalue']] + \
                   ['Intercept coef', 'Intercept tvalue', 'adj_r2', 'n']

        table = table.reindex(index=newIndex)

        table.to_csv(os.path.join(os.path.join(self.path, 'fama macbeth regression analysis.csv')))


#TODO: something wrong with this version,it is obvious with the result in D:\zht\database\quantDb\researchTopics\assetPricing2\size_12M
