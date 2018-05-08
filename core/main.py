# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-28  14:58
# NAME:assetPricing2-main.py
import os
import shutil
import time
import pandas as pd
import numpy as np
from config import WINSORIZE_LIMITS

from data.dataApi import Database, Benchmark
from tool import monitor, summary_statistics, cal_corr, cal_persistence, \
    cal_breakPoints, count_groups, my_average, \
    assign_port_id, famaMacBeth, apply_col_by_col, newey_west, correlation_mixed
from zht.utils.mathu import winsorize, get_outer_frame

DATA=Database(sample_control=True) #TODO: use controlled data
# In the fm function,independent variables are winsorized,so we do not need to filter the raw data.

def combine_with_datalagged(indicators,sample_control=True):
    datalagged=Database(sample_control).by_indicators(indicators + ['weight'])
    datat = Database(sample_control).by_indicators(['stockEretM'])
    '''
    sort the lagged characteristics to construct portfolios
    Notice:
        before shift(1),we must groupby('sid').
    '''
    #TODO:shift(1) we result in the sample loss of first month,upgrade this
    #function to take shift in consideration.
    comb = pd.concat([datalagged.groupby('sid').shift(1), datat],axis=1)
    return comb

@apply_col_by_col
def adjust_with_riskModel(x, riskmodel=None):
    '''
    use risk model to adjust the the alpha,
    the risk model can be None (unadjusted) or one of [capm,ff3,ffc,ff5,hxz4]

    :param x:
    :param riskmodel:one of ['capmM', 'ff3M', 'ffcM', 'ff5M', 'hxz4M']
    :return:
    '''
    lags=5
    d={'capm':'capmM',
       'ff3':'ff3M',
       'ffc':'ffcM',
       'ff5':'ff5M',
       'hxz4':'hxz4M'}

    df = pd.DataFrame(x)
    df.columns = ['y']

    if riskmodel in d.keys():
        '''
        we do not need to shift the time index,the index in df denotes time t+1 (the indicators
        have been shifted forward),so,here the time for Stock excess return is consistent with
        the time for benchmarks.Both of them are from time t+1.
        '''
        bench=Benchmark().by_benchmark(riskmodel)
        df=pd.concat([df,bench],axis=1)
        formula='y ~ '+' + '.join(bench.columns.tolist())
        nw = newey_west(formula, df, lags)

        return nw['Intercept'].rename(index={'coef':'alpha_'+riskmodel,
                                             't': 't_alpha_'+riskmodel})
    else:
        formula='y ~ 1'
        nw = newey_west(formula, df, lags)
        return nw['Intercept'].rename(index={'coef': 'excess return',
                                             't': 't excess return'})

def risk_adjust(panel,riskmodels=None):
    '''
    risk adjusted alpha

    :param panel:
    :return:
    '''
    if not riskmodels:
        riskmodels=[None,'capm','ff3','ffc','ff5','hxz4']

    return pd.concat([adjust_with_riskModel(panel,rm)
                   for rm in riskmodels],axis=0)



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
        corr=correlation_mixed(comb)
        corr.to_csv(os.path.join(self.path, 'corr.csv'))

    @monitor
    def persistence(self):
        # TODO: Table II of Asness, Clifford S., Andrea Frazzini, and Lasse Heje Pedersen. “Quality Minus Junk.” SSRN Scholarly Paper. Rochester, NY: Social Science Research Network, June 5, 2017. https://papers.ssrn.com/abstract=2312432.

        perdf = pd.DataFrame()
        for indicator in self.indicators:
            per = cal_persistence(self.df[indicator].unstack(),
                                  offsets=[1, 3, 6, 12, 24, 36, 48, 60, 120])
            perdf[indicator] = per
        perdf.to_csv(os.path.join(self.path, 'persistence.csv'))

    @monitor
    def breakPoints_and_countGroups(self):
        dfs_bp = []
        dfs_count = []
        for indicator in self.indicators:
            d = self.df[indicator].unstack()
            # there is no samples for some months due to festival
            # TODO: how to set the thresh?
            d = d.dropna(axis=0,thresh=self.q * 10)
            bps = cal_breakPoints(d, self.q)
            dfs_bp.append(bps)
            count = count_groups(d, self.q)
            dfs_count.append(count)
        result_bp = pd.concat(dfs_bp, keys=self.indicators, axis=0)
        result_count = pd.concat(dfs_count, keys=self.indicators, axis=0)

        result_bp.to_csv(os.path.join(self.path, 'breakPoints.csv'))
        result_count.to_csv(os.path.join(self.path, 'count.csv'))

        # TODO:In fact,the count is not exactly the number of stocks to calculate the weighted return
        # TODO:as some stocks will be deleted due to the missing of weights.

    def _get_port_data(self, indicator):
        groupid = DATA.by_indicators([indicator])
        groupid['g'] = groupid.groupby('t', group_keys=False).apply(
            lambda df: pd.qcut(df[indicator], self.q,
                               labels=[indicator + str(i) for i in range(1, self.q + 1)])
        )
        return groupid

    @monitor
    def portfolio_characteristics(self, sortedIndicator, otherIndicators):
        '''
        as table 12.3 panel A
        :param sortedIndicator:
        :param otherIndicators:
        :return:
        '''
        groupid = self._get_port_data(sortedIndicator)
        comb = DATA.by_indicators(otherIndicators)
        comb = pd.concat([groupid, comb], axis=1)
        characteristics_avg = comb.groupby(['t', 'g']).mean().groupby('g').mean()
        characteristics_avg.to_csv(os.path.join(self.path, 'portfolio characteristics.csv'))

    # TODO: upgrade this function
    def _get_panel_stk_avg(self, comb, indicator, gcol):
        panel_stk_eavg = comb.groupby(['t', gcol])['stockEretM'].mean()
        if self.factor == 'size':
            '''
            when the factor is size,we also use the indicator (sort variable) as weight
            Refer to page 159.

            '''
            panel_stk_wavg = comb.groupby(['t', gcol]).apply(
                lambda df: my_average(df, 'stockEretM', wname=indicator)
            )
        else:
            '''
            the index denotes t+1,and the weight is from time t,
            since we have shift weight forward in dataset.
            '''
            # def func(df):
            #     return my_average(df,'stockEretM',wname='weight')
            #
            # panel_stk_wavg=comb.groupby(['t',gcol]).apply(func)
            panel_stk_wavg = comb.groupby(['t', gcol]).apply(
                lambda df: my_average(df, 'stockEretM', wname='weight')
            )

        return panel_stk_eavg, panel_stk_wavg

    @monitor
    def portfolio_analysis(self):
        '''
        table 8.4

        :return:
        '''
        comb=combine_with_datalagged(self.indicators)
        # all_indicators = list(set(self.indicators + ['weight', 'stockEretM']))
        # comb = DATA.by_indicators(all_indicators)

        result_eavg = []
        result_wavg = []
        for indicator in self.indicators:
            gcol = 'g_%s' % indicator
            # comb[gcol]=comb.groupby('t').apply(
            #     lambda df:grouping(df[indicator].reset_index(level='t'),self.q,labels=self.groupnames))
            comb[gcol] = comb.groupby('t', group_keys=False).apply(
                lambda df: assign_port_id(df[indicator], self.q, self.groupnames))
            # TODO:Add an alternative sorting method,that is,updating yearly as page 9 of Chen et al., “On the Predictability of Chinese Stock Returns.”

            panel_stk_eavg, panel_stk_wavg = self._get_panel_stk_avg(comb, indicator, gcol)
            for panel_stk in [panel_stk_eavg, panel_stk_wavg]:
                panel = panel_stk.unstack(level=[gcol])
                panel.columns = panel.columns.astype(str)
                panel['_'.join([self.groupnames[-1], self.groupnames[0]])] = panel[self.groupnames[-1]] - panel[
                    self.groupnames[0]]
                panel['avg'] = panel.mean(axis=1)
                # TODO: use the risk models declared above

                # part A
                a_data = comb.groupby(['t', gcol])[indicator].mean()
                a_data = a_data.unstack()
                a_data.columns = a_data.columns.astype(str)
                a_data.index = a_data.index.astype(str)
                a_data['_'.join([self.groupnames[-1], self.groupnames[0]])] = a_data[self.groupnames[-1]] - a_data[
                    self.groupnames[0]]
                a_data['avg'] = a_data.mean(axis=1)
                a = a_data.mean()
                a.name = 'avg'
                a = a.to_frame().T

                riskAdjusted = risk_adjust(panel)
                # TODO:something must be wrong with size or portfolio_analysse.
                if panel_stk is panel_stk_eavg:
                    result_eavg.append(pd.concat([a, riskAdjusted], axis=0))
                else:
                    result_wavg.append(pd.concat([a, riskAdjusted], axis=0))
        table_e = pd.concat(result_eavg, axis=0, keys=self.indicators)
        table_w = pd.concat(result_wavg, axis=0, keys=self.indicators)
        # reorder the columns
        initialOrder = table_e.columns.tolist()
        h=self.groupnames+['avg']
        newOrder=h+[col for col in initialOrder if col not in h]
        # newOrder = self.groupnames + [col for col in initialOrder if col not in self.groupnames]
        table_e = table_e.reindex(columns=newOrder)
        table_w = table_w.reindex(columns=newOrder)

        #mark the t values to facilitate the following analysis
        table_e['significant_positive']=table_e.iloc[:,-1].map(lambda v:1 if v>2 else np.nan)
        table_e['significant_negative']=table_e.iloc[:,-2].map(lambda v:-1 if v<-2 else np.nan)
        table_w['significant_positive']=table_w.iloc[:,-1].map(lambda v:1 if v>2 else np.nan)
        table_w['significant_negative']=table_w.iloc[:,-2].map(lambda v:-1 if v<-2 else np.nan)

        table_e.to_csv(os.path.join(self.path, 'univariate portfolio analysis-equal weighted.csv'))
        table_w.to_csv(os.path.join(self.path, 'univariate portfolio analysis-value weighted.csv'))

    def _one_indicator(self, indicator):
        ns = range(1, 13)
        all_indicators=[indicator,'weight','stockEretM']
        comb = DATA.by_indicators(all_indicators)
        comb = comb.dropna()
        comb['g'] = comb.groupby('t', group_keys=False).apply(
            lambda df: pd.qcut(df[indicator], self.q,
                               labels=[indicator + str(i) for i in range(1, self.q + 1)])
        )

        def _one_indicator_one_weight_type(group_ts, indicator):
            def _big_minus_small(s, ind):
                time = s.index.get_level_values('t')[0]
                return s[(time, ind + str(self.q))] - s[(time, ind + '1')]

            spread_data = group_ts.groupby('t').apply(lambda series: _big_minus_small(series, indicator))
            s = risk_adjust(spread_data)
            return s

        eret = comb['eret'].unstack()

        s_es = []
        s_ws = []
        eret_names = []
        for n in ns:
            eret_name = 'eret_ahead%s' % (n + 1)
            comb[eret_name] = eret.shift(-n).stack()

            group_eavg_ts = comb.groupby(['t', 'g'])[eret_name].mean()
            group_wavg_ts = comb.groupby(['t', 'g']).apply(
                lambda df: np.average(df[eret_name], weights=df['weight']))
            #TODO: If we are analyzing size,the weights should be the indicator
            #we are analyzing,rather than weight
            s_e = _one_indicator_one_weight_type(group_eavg_ts, indicator)
            s_w = _one_indicator_one_weight_type(group_wavg_ts, indicator)
            s_es.append(s_e)
            s_ws.append(s_w)
            eret_names.append(eret_name)
        eq_table = pd.concat(s_es, axis=1, keys=eret_names)
        vw_table = pd.concat(s_ws, axis=1, keys=eret_names)
        return eq_table, vw_table

    @monitor
    def portfolio_anlayse_with_k_month_ahead_returns(self):
        '''table 11.4'''
        eq_tables = []
        vw_tables = []
        for indicator in self.indicators:
            eq_table, vw_table = self._one_indicator(indicator)
            eq_tables.append(eq_table)
            vw_tables.append(vw_table)
            print(indicator)

        eq = pd.concat(eq_tables, axis=0, keys=self.indicators)
        vw = pd.concat(vw_tables, axis=0, keys=self.indicators)

        eq.to_csv(os.path.join(self.path, 'univariate portfolio analysis_k-month-ahead-returns-eq.csv'))
        vw.to_csv(os.path.join(self.path, 'univariate portfolio analysis_k-month-ahead-returns-vw.csv'))

    @monitor
    def fm(self):
        comb=combine_with_datalagged(self.indicators)
        data = []
        ps = []
        for indicator in self.indicators:
            subdf = comb[[indicator, 'stockEretM']]
            subdf = subdf.dropna()
            subdf.columns = ['y', 'x']
            '''
            (page 141)The independent variable is winsorized at a given level on a monthly basis.
            
            (page 90)The independent variables are usually winsorized to ensure that a small number of extreme
            independent variable values do not have a large effect on the results of the regression.
            
            In some cases the dependent variable is also winsorized.When the dependent variable is a 
            security return or excess return,this variable is usually not winsorized.In most other 
            cases,it is common to winsorized the dependent variable.
            '''
            subdf['x'] = subdf.groupby('t')['x'].apply(lambda s: winsorize(s, limits=WINSORIZE_LIMITS))
            subdf = subdf.reset_index()
            formula = 'y ~ x'
            r, adj_r2, n, p= famaMacBeth(formula, 't', subdf, lags=5)
            # TODO: why intercept tvalue is so large?
            # TODO: why some fm regression do not have a adj_r2 ?
            data.append([r.loc['x', 'coef'], r.loc['x', 'tvalue'],
                         r.loc['Intercept', 'coef'], r.loc['Intercept', 'tvalue'],
                         adj_r2, n])
            ps.append(p['x'])
            print(indicator)

        result = pd.DataFrame(data, index=self.indicators,
                              columns=['slope', 't', 'Intercept', 'Intercept_t', 'adj_r2', 'n']).T
        result.to_csv(os.path.join(self.path, 'fama macbeth regression analysis.csv'))

        parameters = pd.concat(ps, axis=1, keys=self.indicators)
        parameters.to_csv(os.path.join(self.path, 'fama macbeth regression parameters in first stage.csv'))

    @monitor
    def parameter_ts_fig(self):
        '''
        田利辉 and 王冠英, “我国股票定价五因素模型.”

        :return:
        '''
        #TODO: Why not plot rolling parameters?
        parameters = pd.read_csv(os.path.join(self.path, 'fama macbeth regression parameters in first stage.csv'),
                                 index_col=[0], parse_dates=True)
        parameters['zero'] = 0.0
        for indicator in self.indicators:
            s=parameters[indicator].dropna()
            positive_ratio=(s>0).sum()/s.shape[0]
            fig = parameters[[indicator, 'zero']].plot(title='positive ratio: {:.3f}'.format(positive_ratio)).get_figure()
            fig.savefig(os.path.join(self.path, 'fm parameter ts fig-{}.png'.format(indicator)))

    def run(self):
        self.summary()
        self.correlation()
        self.persistence()
        self.breakPoints_and_countGroups()

        # self.portfolio_characteristics()
        self.portfolio_analysis()
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
        comb=combine_with_datalagged([self.indicator1,self.indicator2])
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
        comb=combine_with_datalagged(indicators)
        comb=comb.dropna()
        comb['g1']=comb.groupby('t',group_keys=False).apply(
            lambda df:assign_port_id(df[indicators[0]], self.q,
                                     [indicators[0] + str(i) for i in range(1,self.q + 1)]))

        comb['g2']=comb.groupby(['t','g1'],group_keys=False).apply(
            lambda df:assign_port_id(df[indicators[1]], self.q,
                                     [indicators[1] + str(i) for i in range(1,self.q + 1)]))

        return comb

    def _get_eret(self,comb):
        group_eavg_ts = comb.groupby(['g1', 'g2', 't'])['stockEretM'].mean()
        group_wavg_ts = comb.groupby(['g1', 'g2', 't']).apply(
            lambda df: np.average(df['stockEretM'], weights=df['weight']))
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

    @staticmethod
    def famaMacbeth_reg(indeVars):
        #TODO: upgrade fm in Univariate by calling this function
        '''
        (page 141)The independent variable is winsorized at a given level on a monthly basis.

        (page 90)The independent variables are usually winsorized to ensure that a small number of extreme
        independent variable values do not have a large effect on the results of the regression.

        In some cases the dependent variable is also winsorized.When the dependent variable is a
        security return or excess return,this variable is usually not winsorized.In most other
        cases,it is common to winsorized the dependent variable.
        '''
        comb=combine_with_datalagged(indeVars)
        comb=comb.dropna()

        # winsorize
        comb[indeVars]=comb.groupby('t')[indeVars].apply(
            lambda x:winsorize(x,limits=WINSORIZE_LIMITS,axis=0))
        namedict={inde:'name{}'.format(i) for i,inde in enumerate(indeVars)}
        comb=comb.rename(columns=namedict)
        formula = 'stockEretM ~ ' + ' + '.join(namedict.values())
        # TODO:lags?
        r, adj_r2, n, firstStage_params = famaMacBeth(formula, 't', comb, lags=5)  # TODO:
        r = r.rename(index={v:k for k,v in namedict.items()})
        # save the first stage regression parameters
        firstStage_params = firstStage_params.rename(
            columns={v:k for k,v in namedict.items()})
        params = r[['coef', 'tvalue']].stack()
        params.index = params.index.map('{0[0]} {0[1]}'.format)
        params['adj_r2'] = adj_r2
        params['n'] = n
        return params,firstStage_params

    def _fm(self,x):
        '''

        :param x: a list of list,or just a list contains the name of independent variables
        :return:
        '''
        if isinstance(x[0],str):
            p, firstStage_params = self.famaMacbeth_reg(x)
            firstStage_params.to_csv(os.path.join(self.path, 'first stage parameters ' + '_'.join(x) + '.csv'))
            p.to_csv(os.path.join(os.path.join(self.path,'fama macbeth regression analysis.csv')))
        if isinstance(x[0],list):
            ps=[]
            for indeVars in x:
                p,firstStage_params=self.famaMacbeth_reg(indeVars)
                firstStage_params.to_csv(os.path.join(self.path, 'first stage parameters ' + '_'.join(indeVars) + '.csv'))
                ps.append(p)
            table = pd.concat(ps, axis=1, keys=range(1, len(x) + 1))
            all_indeVars = list(set(var for l_indeVars in x for var in l_indeVars))
            newIndex = [var + ' ' + suffix for var in all_indeVars for suffix in ['coef', 'tvalue']] + \
                       ['Intercept coef', 'Intercept tvalue', 'adj_r2', 'n']
            table = table.reindex(index=newIndex)
            table.to_csv(os.path.join(os.path.join(self.path, 'fama macbeth regression analysis.csv')))

#TODO: wrong!!!! For predictors with accounting data updated annually