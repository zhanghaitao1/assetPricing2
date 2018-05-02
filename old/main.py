# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-22  15:09
# NAME:assetPricing2-main.py
import shutil

import time

from config import WINSORIZE_LIMITS
from dataset import DATA, BENCH
from tool import assign_port_id, monitor, apply_col_by_col, my_average, summary_statistics, \
    cal_breakPoints, count_groups, famaMacBeth, newey_west, cal_corr, cal_persistence

import os
import pandas as pd
import numpy as np
from zht.utils.mathu import get_outer_frame, winsorize


@apply_col_by_col
def adjust_with_riskModel(x, riskmodel=None):
    '''
    use risk model to adjust the the alpha,
    the risk model can be None (unadjusted) or one of [capm,ff3,ffc,ff5,hxz4]

    :param x:
    :param riskmodel:
    :return:
    '''
    lags=5
    d={'capm':'rpM',
       'ff3':'ff3M',
       'ffc':'ffcM',
       'ff5':'ff5M',
       'hxz4':'hxz4M'}

    df = pd.DataFrame(x)
    df.columns = ['y']

    if riskmodel in d.keys():
        bench=BENCH.data[riskmodel]
        df=df.join(bench)
        formula='y ~ '+' + '.join(bench.columns.tolist())
        nw = newey_west(formula, df, lags)

        # return nw['Intercept'].rename(index={'coef': riskmodel+'_alpha',
        #                                      't': riskmodel+'_alpha_t'})

        return nw['Intercept'].rename(index={'coef':'alpha_'+riskmodel,
                                             't': 't_alpha'+riskmodel})
    else:
        formula='y ~ 1'
        nw = newey_west(formula, df, lags)
        return nw['Intercept'].rename(index={'coef': 'excess return',
                                             't': 't excess return'})

def risk_adjust(panel):
    '''
    risk adjusted alpha

    :param panel:
    :return:
    '''
    return pd.concat([adjust_with_riskModel(panel,riskmodel)
                   for riskmodel in [None,'capm','ff3','ffc','ff5','hxz4']],
                  axis=0)


class OneFactor:
    q=10

    def __init__(self, factor,path):
        self.factor=factor
        self.path=path
        self.indicators=DATA.info[factor]
        self.df=DATA.data[self.indicators]
        self.groupnames=[self.factor+str(i) for i in range(1,self.q+1)]
        self._build_environment()

    def _build_environment(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
            time.sleep(0.1)
        os.makedirs(self.path)


    @monitor
    def summary(self):
        series=[]
        for indicator in self.indicators:
            s=summary_statistics(self.df[indicator].unstack())
            series.append(s.mean())
        pd.concat(series,keys=self.indicators,axis=1).to_csv(os.path.join(self.path,'summary.csv'))

    @monitor
    def correlation(self,indicators=None):
        if not indicators:
            indicators=self.indicators

        comb=DATA.by_indicators(indicators)
        def _spearman(df):
            df=df.dropna()
            if df.shape[0]>10:#TODO:thresh to choose
                return cal_corr(df,'spearman',winsorize=False)

        def _pearson(df):
            df=df.dropna()
            if df.shape[0]>10:#TODO: min_samples
                return cal_corr(df,'pearson',winsorize=True)


        corrs=comb.groupby('t').apply(_spearman)
        corrp=comb.groupby('t').apply(_pearson)

        corrsAvg=corrs.groupby(level=1).mean().reindex(index=indicators, columns=indicators)
        corrpAvg=corrp.groupby(level=1).mean().reindex(index=indicators, columns=indicators)

        corr1 = np.tril(corrpAvg.values, k=-1)
        corr2 = np.triu(corrsAvg.values, k=1)

        corr = pd.DataFrame(corr1 + corr2, index=corrpAvg.index, columns=corrpAvg.columns)
        np.fill_diagonal(corr.values, np.NaN)
        corr.to_csv(os.path.join(self.path, 'corr.csv'))

    @monitor
    def persistence(self):
        #TODO: Table II of Asness, Clifford S., Andrea Frazzini, and Lasse Heje Pedersen. “Quality Minus Junk.” SSRN Scholarly Paper. Rochester, NY: Social Science Research Network, June 5, 2017. https://papers.ssrn.com/abstract=2312432.

        perdf=pd.DataFrame()
        for indicator in self.indicators:
            per=cal_persistence(self.df[indicator].unstack(),
                                         offsets=[1, 3, 6, 12, 24, 36, 48, 60, 120])
            perdf[indicator]=per
        perdf.to_csv(os.path.join(self.path,'persistence.csv'))

    @monitor
    def breakPoints_and_countGroups(self):
        dfs_bp=[]
        dfs_count=[]
        for indicator in self.indicators:
            d=self.df[indicator].unstack()
            #there is no samples for some months due to festival
            #TODO: how to set the thresh?
            d=d.dropna(axis=0,how='all',thresh=self.q*10)
            bps=cal_breakPoints(d,self.q)
            dfs_bp.append(bps)
            count=count_groups(d,self.q)
            dfs_count.append(count)
        result_bp=pd.concat(dfs_bp,keys=self.indicators,axis=0)
        result_count=pd.concat(dfs_count,keys=self.indicators,axis=0)

        result_bp.to_csv(os.path.join(self.path,'breakPoints.csv'))
        result_count.to_csv(os.path.join(self.path,'count.csv'))

        # TODO:In fact,the count is not exactly the number of stocks to calculate the weighted return
        # TODO:as some stocks will be deleted due to the missing of weights.

    def _get_port_data(self,indicator):
        groupid=DATA.by_indicators([indicator])
        groupid['g']=groupid.groupby('t',group_keys=False).apply(
            lambda df:pd.qcut(df[indicator],self.q,
                              labels=[indicator+str(i) for i in range(1,self.q+1)])
        )
        return groupid

    @monitor
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
            '''
            when the factor is size,we also use the indicator (sort variable) as weight
            Refer to page 159.
            
            '''
            panel_stk_wavg=comb.groupby(['t',gcol]).apply(
                lambda df:my_average(df,'eretM',wname=indicator)
                )
        else:
            '''
            the index denotes t+1,and the capM is from time t,
            since we have shift capM forward in dataset.
            '''
            panel_stk_wavg = comb.groupby(['t', gcol]).apply(
                lambda df:my_average(df,'eretM',wname='capM')
            )

        return panel_stk_eavg,panel_stk_wavg

    @monitor
    def portfolio_analysis(self):
        '''
        table 8.4

        :return:
        '''
        #TODO: add a parameter to declare what risk models will be used. [ff3,capm,ff5]

        all_indicators = list(set(self.indicators + ['capM', 'eretM']))
        comb = DATA.by_indicators(all_indicators)

        result_eavg=[]
        result_wavg=[]
        for indicator in self.indicators:
            gcol='g_%s'%indicator
            # comb[gcol]=comb.groupby('t').apply(
            #     lambda df:grouping(df[indicator].reset_index(level='t'),self.q,labels=self.groupnames))
            comb[gcol]=comb.groupby('t',group_keys=False).apply(
                lambda df:assign_port_id(df[indicator], self.q, self.groupnames))
            #TODO:Add an alternative sorting method,that is,updating yearly as page 9 of Chen et al., “On the Predictability of Chinese Stock Returns.”

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

                riskAdjusted=risk_adjust(panel)
                #TODO:something must be wrong with size or portfolio_analysse.
                if panel_stk is panel_stk_eavg:
                    result_eavg.append(pd.concat([a,riskAdjusted],axis=0))
                else:
                    result_wavg.append(pd.concat([a,riskAdjusted],axis=0))
        table_e=pd.concat(result_eavg,axis=0,keys=self.indicators)
        table_w=pd.concat(result_wavg,axis=0,keys=self.indicators)
        #reorder the columns
        initialOrder=table_e.columns.tolist()
        newOrder=self.groupnames+[col for col in initialOrder if col not in self.groupnames]
        table_e=table_e.reindex(columns=newOrder)
        table_w=table_w.reindex(columns=newOrder)

        table_e.to_csv(os.path.join(self.path,'univariate portfolio analysis-equal weighted.csv'))
        table_w.to_csv(os.path.join(self.path,'univariate portfolio analysis-value weighted.csv'))

    def _one_indicator(self,indicator):
        ns=range(0,12)
        all_indicators=list(set([indicator]+['capM','eretM']))
        comb=DATA.get_by_indicators(all_indicators)
        comb=comb.dropna()
        comb['g']=comb.groupby('t',group_keys=False).apply(
            lambda df:pd.qcut(df[indicator],self.q,
                              labels=[indicator+str(i) for i in range(1,self.q+1)])
        )

        def _one_indicator_one_weight_type(group_ts, indicator):
            def _big_minus_small(s, ind):
                time=s.index.get_level_values('t')[0]
                return s[(time, ind + str(self.q))] - s[(time, ind + '1')]

            spread_data=group_ts.groupby('t').apply(lambda series:_big_minus_small(series, indicator))
            s=risk_adjust(spread_data)
            return s

        eret=comb['eret'].unstack()

        s_es=[]
        s_ws=[]
        eret_names=[]
        for n in ns:
            eret_name='eret_ahead%s'%(n+1)
            comb[eret_name]=eret.shift(-n).stack()

            group_eavg_ts=comb.groupby(['t','g'])[eret_name].mean()
            group_wavg_ts=comb.groupby(['t','g']).apply(lambda df:np.average(df[eret_name],weights=df['mktCap']))

            s_e=_one_indicator_one_weight_type(group_eavg_ts,indicator)
            s_w=_one_indicator_one_weight_type(group_wavg_ts,indicator)
            s_es.append(s_e)
            s_ws.append(s_w)
            eret_names.append(eret_name)
        eq_table=pd.concat(s_es,axis=1,keys=eret_names)
        vw_table=pd.concat(s_ws,axis=1,keys=eret_names)
        return eq_table,vw_table

    @monitor
    def portfolio_anlayse_with_k_month_ahead_returns(self):
        '''table 11.4'''
        eq_tables=[]
        vw_tables=[]
        for indicator in self.indicators:
            eq_table,vw_table=self._one_indicator(indicator)
            eq_tables.append(eq_table)
            vw_tables.append(vw_table)
            print(indicator)

        eq=pd.concat(eq_tables,axis=0,keys=self.indicators)
        vw=pd.concat(vw_tables,axis=0,keys=self.indicators)

        eq.to_csv(os.path.join(self.path,'univariate portfolio analysis_k-month-ahead-returns-eq.csv'))
        vw.to_csv(os.path.join(self.path,'univariate portfolio analysis_k-month-ahead-returns-vw.csv'))

    @monitor
    def fm(self,wsz=None):
        comb=DATA.by_indicators(self.indicators+['eretM'])
        data=[]
        ps=[]
        for indicator in self.indicators:
            subdf=comb[[indicator,'eretM']]
            subdf=subdf.dropna()
            subdf.columns=['y','x']
            # The independent variable is winsorized at a given level on a monthly basis. as page 141
            subdf['x']=subdf.groupby('t')['x'].apply(lambda s:winsorize(s,limits=WINSORIZE_LIMITS))
            subdf=subdf.reset_index()
            formula='y ~ x'
            r,adj_r2,n,p=famaMacBeth(formula,'t',subdf,lags=5)
            #TODO: why intercept tvalue is so large?
            # TODO: why some fm regression do not have a adj_r2 ?
            data.append([r.loc['x', 'coef'], r.loc['x', 'tvalue'],
                         r.loc['Intercept', 'coef'], r.loc['Intercept', 'tvalue'],
                         adj_r2, n])
            ps.append(p['x'])
            print(indicator)

        result = pd.DataFrame(data, index=self.indicators,
                              columns=['slope', 't', 'Intercept', 'Intercept_t', 'adj_r2', 'n']).T
        result.to_csv(os.path.join(self.path, 'fama macbeth regression analysis.csv'))

        parameters=pd.concat(ps,axis=1,keys=self.indicators)
        parameters.to_csv(os.path.join(self.path,'fama macbeth regression parameters in first stage.csv'))

    @monitor
    def parameter_ts_fig(self):
        '''
        田利辉 and 王冠英, “我国股票定价五因素模型.”

        :return:
        '''
        parameters=pd.read_csv(os.path.join(self.path,'fama macbeth regression parameters in first stage.csv'),
                               index_col=[0],parse_dates=True)
        parameters['zero']=0.0
        for indicator in self.indicators:
            fig=parameters[[indicator,'zero']].plot().get_figure()
            fig.savefig(os.path.join(self.path,'fm parameter ts fig-{}.png'.format(indicator)))


    def run(self):
        # self.summary()
        # self.correlation()
        # self.persistence()
        # self.breakPoints_and_countGroups()

        ## self.portfolio_characteristics()
        # self.portfolio_analysis()
        self.fm()
        self.parameter_ts_fig()


    def __call__(self):
        self.run()

class Bivariate:
    q=5
    def __init__(self,indicator1,indicator2,proj_path):
        self.indicator1=indicator1
        self.indicator2=indicator2
        self.path=proj_path
        self._build_environment()

    def _build_environment(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path)

    def _get_independent_data(self):
        # TODO: add the method of ratios such as [0.3,0.7]
        # sometimes the self.indicators and ['capM','eretM'] may share some elements
        comb=DATA.by_indicators([self.indicator1,self.indicator2,'capM','eretM'])
        comb=comb.dropna()
        comb['g1']=comb.groupby('t',group_keys=False).apply(
            lambda df:assign_port_id(df[self.indicator1], self.q,
                                     [self.indicator1 + str(i) for i in range(1, self.q + 1)]))

        comb['g2']=comb.groupby('t',group_keys=False).apply(
            lambda df:assign_port_id(df[self.indicator2], self.q,
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
            lambda df:assign_port_id(df[indicators[0]], self.q,
                                     [indicators[0] + str(i) for i in range(1,self.q + 1)]))

        comb['g2']=comb.groupby(['t','g1'],group_keys=False).apply(
            lambda df:assign_port_id(df[indicators[1]], self.q,
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

    @monitor
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

    @monitor
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

    @monitor
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
        indeVars=list(set(var for l_indeVars in ll_indeVars for var in l_indeVars))
        indicators = indeVars + ['eretM']
        comb = DATA.by_indicators(indicators)
        # The independent variable is winsorized at a given level on a monthly basis. as page 170
        comb[indeVars]=comb.groupby('t')[indeVars].apply(lambda x:winsorize(x,limits=WINSORIZE_LIMITS,axis=0))
        comb = comb.reset_index()
        stks = []
        for l_indeVars in ll_indeVars:
            '''
            replace the old name with new name,since patsy do not support name starts with number 

            '''
            newname = ['name' + str(i) for i in range(1, len(l_indeVars) + 1)]
            df = comb[l_indeVars + ['t', 'eretM']].dropna()
            df.columns = newname + ['t', 'eretM']
            formula = 'eretM ~ ' + ' + '.join(newname)
            # TODO:lags?
            r, adj_r2, n,p= famaMacBeth(formula, 't', df, lags=5)#TODO:
            r = r.rename(index=dict(zip(newname, l_indeVars)))
            #save the first stage regression parameters
            p=p.rename(columns=dict(zip(newname,l_indeVars)))
            p.to_csv(os.path.join(self.path,'first stage parameters '+'_'.join(l_indeVars)+'.csv'))
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
