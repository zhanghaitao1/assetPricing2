# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-19  13:44
# NAME:assetPricing2-din.py

import pandas as pd
import numpy as np
import multiprocessing
from data.dataTools import read_df_from_gta, save, \
    read_gta, read_unfiltered, quaterly2monthly
from zht.data.resset.api import read_resset
from zht.utils.dateu import freq_end
from zht.data.wind.api import read_wind
from zht.utils.mathu import get_inter_frame


def get_stockRetD():
    # get stock daily stock return
    tbname='TRD_Dalyr'
    varname='Dretwd'#考虑现金红利再投资的收益
    indname='Trddt'
    colname='Stkcd'
    df=read_df_from_gta(tbname, varname, indname, colname)

    df.index.name='t'
    df.index=pd.to_datetime(df.index) #TODO: dayend?
    df.columns.name='sid'
    df.columns=df.columns.astype(str)

    save(df, 'stockRetD')
    return df

def get_stockCloseD():
    #get daily stock close price
    tbname='TRD_Dalyr'
    varname='Adjprcwd'# adjusted close price with dividend taken into consideration
    indname='Trddt'
    colname='Stkcd'
    df=read_df_from_gta(tbname, varname, indname, colname)

    df.index.name='t'
    df.index=pd.to_datetime(df.index)
    df.columns.name='sid'
    df.columns=df.columns.astype(str)

    df=df.sort_index(axis=0)
    df=df.sort_index(axis=1)

    save(df, 'stockCloseD')
    return df

def get_mktRetD():
    # get daily market return

    tbname = 'TRD_Cndalym'
    indVar = 'Trddt'
    targetVar = 'Cdretwdos'  #trick 考虑现金红利再投资的综合日市场回报率(流通市值加权平均法)
    df=read_gta(tbname)


    condition1=df['Markettype']==21 #trick 21=综合A股和创业板
    df=df[condition1]

    df = df.set_index(indVar)
    df.index.name='t'
    df.index = pd.to_datetime(df.index)

    s = df[targetVar]
    s.name='mktRetD' #TODO: put this line into check function or unify function?
    save(s, 'mktRetD')

def _get_rf(freq):
    '''
    parse risk free rate from the database
    Args:
        freq: D (daily),W (weekly),M (monthly)

    Returns:

    '''
    dic={'D':'Nrrdaydt','W':'Nrrwkdt','M':'Nrrmtdt'}

    tname = 'TRD_Nrrate'
    src = read_gta(tname)
    #NRI01=定期-整存整取-一年利率；TBC=国债票面利率,根据复利计算方法，
    # 将年度的无风险利率转化为月度数据
    src=src[src['Nrr1']=='NRI01']#trick: choose the type of risk free rate
    src=src.set_index('Clsdt')

    rf=src[dic[freq]][2:]#delete the first two rows
    rf.index.name='t'
    rf.name='rf'+freq

    rf.index=pd.to_datetime(rf.index)
    if freq in ['W','M']:
        rf=rf.resample(freq).agg(lambda x:x[round(x.shape[0]/2)])

    return rf/100.0 #trick:the unit of rf in the file is %,we adjust it to be actual value.

def get_rfD():
    df=_get_rf('D')
    save(df,'rfD')
    return df

def get_rfM():
    df=_get_rf('M')
    save(df,'rfM')
    return df

def get_stockEretD():
    stockRetD=get_stockRetD()
    rfD=get_rfD()
    stockEretD=stockRetD.sub(rfD,axis=0)
    # The date for stockRetD is buisiness date,but for rfD, it is calendar date.
    stockEretD=stockEretD.dropna(axis=0,how='all')# use this to ajust the index from calendar date to buisiness date

    save(stockEretD,'stockEretD')
    return stockEretD

def get_stockRetM():
    '''
    monthly stock return with dividend
    '''
    tbname = 'TRD_Mnth'
    varname='Mretwd'#考虑现金红利再投资的收益
    indname='Trdmnt'
    colname='Stkcd'

    df=read_df_from_gta(tbname, varname, indname, colname)

    #TODO: identify the axis and convert the axis automatically
    df.index.name='t'
    df.columns.name='sid'
    df.index=freq_end(df.index, 'M')
    df.columns=df.columns.astype(str)

    save(df, 'stockRetM')
    return df

def get_stockCloseM():
    closeD=get_stockCloseD()
    closeM=closeD.resample('M').last()
    save(closeM,'stockCloseM')
    return closeM

# def get_stockCloseM_old():#debug: this may be wrong, it may not be adjusted close price
#     '''
#     monthly stock close price
#     :return:
#     '''
#     tbname = 'TRD_Mnth'
#     varname='Mclsprc'
#     indname='Trdmnt'
#     colname='Stkcd'
#
#     df=read_df_from_gta(tbname, varname, indname, colname)
#
#     #TODO: identify the axis and convert the axis automatically
#     df.index.name='t'
#     df.columns.name='sid'
#     df.index=freq_end(df.index, 'M')
#     df.columns=df.columns.astype(str)
#
#     save(df, 'stockCloseM')

def get_stockEretM():
    stockRetM=get_stockRetM()
    rfM=get_rfM()
    stockEretM = stockRetM.sub(rfM, axis=0)
    save(stockEretM, 'stockEretM')

def get_mktRetM():
    tbname = 'TRD_Cnmont'
    indVar = 'Trdmnt'
    targetVar = 'Cmretwdos'  #trick:考虑现金红利再投资的综合日市场回报率(流通市值加权平均法)

    df = read_gta(tbname)
    df=df[df['Markettype']==21]# 21=综合A股和创业板

    df = df.set_index(indVar)
    df.index=freq_end(df.index, 'M')
    df.index.name='t'

    s = df[targetVar]
    s.name='mktRetM'

    save(s,'mktRetM')

def get_capM():
    '''
    get stock monthly circulation market capitalization

    :return:
    '''
    tbname='TRD_Mnth'
    varname='Msmvosd' #trick:月个股流通市值，单位 千元
    # TODO:the unit convert it to million as Cakici, Chan, and Topyan, “Cross-Sectional Stock Return Predictability in China.”
    indname='Trdmnt'
    colname='Stkcd'
    df=read_df_from_gta(tbname, varname, indname, colname)
    df.index.name='t'
    df.index=freq_end(df.index, 'M')
    df.columns=df.columns.astype(str)
    df.columns.name='sid'
    save(df,'capM')

# financial indicators-------------------------------------------
def get_bps():
    tbname = 'FI_T9'
    varname = 'F091001A' # 每股净资产
    indname = 'Accper'
    colname = 'Stkcd'
    df=read_df_from_gta(tbname, varname, indname, colname)
    df.index.name='t'
    df.index=pd.to_datetime(df.index)
    df.columns=df.columns.astype(str)
    df.columns.name='sid'
    save(df,'bps')

def get_bps_wind():
    '''
    from code generator by use

    w.wsd("000001.SZ,000002.SZ,000004.SZ,000005.SZ,000006.SZ", "bps", "2017-02-04", "2018-03-05", "currencyType=;Period=Q;Fill=Previous")

    :return:
    '''
    df=read_wind('bps', freq='M')
    df.index.name='t'
    df.columns.name='sid'

    save(df,'bps_wind')

def get_stockCloseY():
    closeD=get_stockCloseD()
    closeY=closeD.resample('Y').last()
    save(closeY,'stockCloseY')
    return closeY

#get stock close price yearly
# def get_stockCloseY_old():
#     tbname='TRD_Year'
#     varname='Yclsprc' #fixme: this is not adjusted price
#     indname='Trdynt'
#     colname='Stkcd'
#     df=read_df_from_gta(tbname, varname, indname, colname)
#     df.index=freq_end(df.index,'Y')
#     df.index.name='t'
#     df.columns=df.columns.astype(str)
#     df.columns.name='sid'
#
#     save(df,'stockCloseY')

def get_ff3M_resset():
    '''
    from resset data

    :return:
    '''
    tbname='THRFACDAT_MONTHLY'
    df=read_resset(tbname)
    # 'Exchflg == 0'   所有交易所
    # 'Mktflg == A'    只考虑A股
    df=df[(df['Exchflg']==0) & (df['Mktflg']=='A')]
    df=df.set_index('Date')
    df.index=freq_end(df.index,'M')
    df.index.name='t'
    df=df[['Rmrf_tmv','Smb_tmv','Hml_tmv']]#weighted with tradable capitalization
    df.columns=['rp','smb','hml']
    df.columns.name='type'
    save(df,'ff3M_resset')

def get_ff3M():#fixme: there are some abnormal values
    df=read_gta('STK_MKT_ThrfacMonth')
    #trick:P9709 全部A股市场包含沪深A股和创业板
    #trick:流通市值加权
    df=df[df['MarkettypeID']=='P9709'][['TradingMonth','RiskPremium1','SMB1','HML1']]
    df.columns=['t','rp','smb','hml']
    df=df.set_index('t')
    df.index=freq_end(df.index,'M')
    df.columns.name='type'
    save(df,'ff3M')
    return df

def get_ffcM():
    df=read_gta('STK_MKT_CarhartFourFactors')
    #trick: P9709 全部A股市场包含沪深A股和创业板
    #trick: 流通市值加权
    df = df[df['MarkettypeID'] == 'P9709'][
        ['TradingMonth', 'RiskPremium1', 'SMB1', 'HML1', 'UMD2']]
    df.columns = ['t', 'rp', 'smb', 'hml', 'mom']
    df.columns.name='type'
    df = df.set_index('t')
    df.index=freq_end(df.index,'M')

    save(df,'ffcM')

def get_ff5M():
    df=read_gta('STK_MKT_FivefacMonth')
    #trick:P9709 全部A股市场包含沪深A股和创业板
    #trick:流通市值加权
    #trick: 2*3 投资组合
    df=df[(df['MarkettypeID']=='P9709') & (df['Portfolios']==1)][
        ['TradingMonth','RiskPremium1','SMB1','HML1','RMW1','CMA1']]
    df.columns=['t','rp','smb','hml','rmw','cma']
    df.columns.name='type'
    df=df.set_index('t')
    df.index=freq_end(df.index,'M')
    # df.index.name='t'
    save(df,'ff5M')

# def get_hxz4M():
#     '''
#     D:\app\python27\zht\researchTopics\assetPricing\calFactors.py\get_hxz4Factors()
#
#     :return:
#     '''
#
#     fp=r'D:\zht\database\quantDb\researchTopics\assetPricing\benchmarkModel\hxz4.csv'
#     df=pd.read_csv(fp,index_col=0)
#     df.index=freq_end(df.index,'M')
#     df.index.name='t'
#     df.columns.name='type'
#     save(df,'hxz4M')

def get_ff3D():
    tbname='STK_MKT_ThrfacDay'
    df=read_gta(tbname)
    condition1=df['MarkettypeID']=='P9707'
    # P9709 全部A股市场包含沪深A股和创业板.
    # 流通市值加权
    df = df[condition1][
        ['TradingDate', 'RiskPremium1', 'SMB1', 'HML1']]
    df.columns = ['t', 'rp', 'smb', 'hml']
    df.columns.name='type'
    df = df.set_index('t')
    df.index=freq_end(df.index,'D')
    save(df,'ff3D')
    return df

def get_rpM():
    ff3M=get_ff3M()
    rpM=ff3M['rp']
    rpM.name='rpM'
    save(rpM,'rpM')

def get_rpD():
    ff3D=get_ff3D()
    rpD=ff3D['rp']
    rpD.name='rpD'
    save(rpD,'rpD')

def get_capmM():
    ff3M=get_ff3M()
    rpM=ff3M['rp']
    save(rpM,'capmM',axis_info=False)

def get_listInfo():
    df=read_gta('IPO_Cobasic', encoding='gbk')
    df=df.set_index('Stkcd')
    df.index.name='sid'
    df.index=df.index.astype(str)
    df.columns.name='type'
    #TODO: refer to page12 of 动量因子_164.pdf   ' 1 代表剔除金融、保险、 ST 类股票'
    df['not_financial']=df['Indcd']!=1 #financial stocks
    df['not_cross']=~df['Crcd'].notnull()#cross means stocks listed on multiple stock markets
    df['is_sh']=df['Listexg']==1#listed on shanghai
    df['is_sz']=df['Listexg']==2#listed on shenzhen
    # Listdt denotes listed date ,'Ipodt' denotes IPO date
    df['Listdt']=df['Listdt'].replace(['0000-00-00','2100-01-01'],np.nan) #there is some invalid data in column 'Listdt
    df['listDate']=pd.to_datetime(df['Listdt'])
    df=df[['listDate','not_financial','not_cross','is_sh','is_sz']]
    df=df[~df.index.duplicated(False)] #there are some duplicated items such as '600018
    df=df.dropna()
    save(df,'listInfo')

def get_tradingStatusD():
    df=read_gta('TRD_Dalyr', encoding='gbk')
    #Trick: Trdsta==1 means "正常交易"
    df['is_normal']=df['Trdsta']==1.0
    df['t']=pd.to_datetime(df['Trddt'])
    df['sid']=df['Stkcd'].astype(str)
    status=pd.pivot_table(df,values='is_normal',index='t',columns='sid')
    save(status,'tradingStatusD')
    return status

def get_tradingStatusM():
    statusD=get_tradingStatusD()
    statusM=statusD.resample('M').last()
    save(statusM,'tradingStatusM')
    return statusM

# def get_stInfo_old():
#     '''
#     for freq='M',delete all the months as long as ST or *ST appear in any day of that month,
#     for freq='D',we only delete the current day with ST or *ST
#     :return:
#     '''
#     #TODO: how about PT
#     df=read_gta('TRD_Dalyr', encoding='gbk')
#     df=df[['Trddt','Stkcd','Trdsta']]
#     df.columns=['t','sid','status']
#     df.columns.name='type'
#     df['t']=pd.to_datetime(df['t'])
#     df['sid']=df['sid'].astype(str)
#
#     #Trick: only get "正常交易"
#     df0=df[df['status']==1.0].copy() #TODO:or just delete all the ST stocks rather than just delete the corresponding months or days
#     df0['not_st']=True
#     dfD=df0.set_index(['t','sid'])['not_st']
#     dfD=dfD.sort_index(level='t')
#     dfD=dfD.unstack()
#
#     def func(x):
#         # for information about the status refer to the documents
#         result= (2.0 not in x['status'].values) & (3.0 not in x['status'].values)
#         return result
#
#     #delete st months for each stock
#     df1=df.groupby([pd.Grouper(key='t',freq='M'),'sid']).filter(func)
#     dfM=df1.groupby([pd.Grouper(key='t',freq='M'),'sid']).sum()
#     dfM['not_st']=True
#     dfM=dfM['not_st']
#     dfM=dfM.unstack()
#
#     save(dfD,'stInfoD')
#     save(dfM,'stInfoM')

def get_pu():
    '''
    policy uncertainty
    :return:
    '''
    url = r'http://www.policyuncertainty.com/media/China_Policy_Uncertainty_Data.xlsx'
    pu = pd.read_excel(url, skip_footer=1)
    pu.columns = ['year', 'month', 'pu']
    pu['t'] = pu['year'].map(str) + '-' + pu['month'].map(str)
    pu['t'] = freq_end(pu['t'], 'M')
    pu = pu.set_index('t')
    pu = pu['pu']
    save(pu,'pu')

def parse_financial_report(tbname, varname, freq='Y', consolidated=True):
    '''
    This function will parse indicator from financial report.

    Args:
        tbname:
        varname:
        freq:{'Y','Q'},'Y' means yearly,'Q' means quartly
        consolidated: True or False,If true,use the indicator from consolidated
        financial statements

    Returns:DataFrame

    '''
    df=read_gta(tbname)
    if consolidated:
        df=df[df['Typrep']=='A']# 合并报表

    colname='Stkcd'
    indname='Accper'
    table=pd.pivot_table(df,varname,indname,colname)
    table.index.name='t'
    table.index=pd.to_datetime(table.index)
    table.columns=table.columns.astype(str)
    table.columns.name='sid'
    if freq=='Y':
        return table[table.index.month==12]
    elif freq=='Q':
        return table[table.index.month.isin([3,6,9,12])]



'''
refer to this link for details about the relationship between income,
operating income,and so on.
'''

def get_op():
    '''
    calculate operating profitability as in FF5

    Returns:

    '''

    # --------------operating probability---------------
    tbname = 'FS_Comins'
    # var1='B001101000' # 营业收入
    # var2='B001201000' # 营业成本
    # var3='B001209000' # 销售费用
    # var4='B001210000' # 管理费用
    # var5='B001211000' # 财务费用
    var = 'B001300000'  # 营业利润
    # var7='Bbd1102203' # 利息支出
    OP=parse_financial_report(tbname,var)

    # ----------------book value---------------
    tbname = 'FS_Combas'
    # var1 = 'A003000000'  # 所有者权益合计
    var = 'A003100000'  # 归属于母公司所有者权益合计
    BV=parse_financial_report(tbname,var)
    BV[BV<=0]=np.nan #Trick: delete those samples with a negative denominator
    OP, BV = get_inter_frame([OP, BV])
    op = OP / BV
    op.index.name='t'
    op.columns.name='sid'
    op=quaterly2monthly(op)
    save(op,'op')

def get_inv():
    '''
    calculate the growth of total asset:
        I/A in Hou, Xue, and Zhang, “Digesting Anomalies.”
        inv in Fama and French, “A Five-Factor Asset Pricing Model.”

    this indicator is the same as ROE calculated as follows:
        tbname='FI_T8'
        varname='F080602A'
        roe=parse_financial_report(tbname, varname)
        roe=quaterly2monthly(roe)

    References:
        Hou, K., Xue, C., and Zhang, L. (2014). Digesting Anomalies: An Investment Approach. Review of Financial Studies 28, 650–705.

    Returns:

    '''
    tbname='FS_Combas'
    #book value
    varname='A001000000' # 总资产
    ta=parse_financial_report(tbname, varname)
    ta[ta<=0]=np.nan#trick: delete samples with negative denominator
    inv=ta.pct_change()
    inv=quaterly2monthly(inv)
    save(inv,'inv')

#fixme: sometimes, we may use yearly data, and sometimes we use quarterly data, is that right?

def get_roe():
    '''
    roe in HXZ

    References:
        Hou, K., Xue, C., and Zhang, L. (2014). Digesting Anomalies:An Investment Approach. Review of Financial Studies 28, 650–705.


    Returns:

    '''
    tbname1='FS_Comins'
    varname1='B002000000' # 净利润

    income=parse_financial_report(tbname1,varname1,freq='Q')

    tbname2 = 'FS_Combas'
    # var1 = 'A003000000'  # 所有者权益合计
    var2 = 'A003100000'  # 归属于母公司所有者权益合计
    BV = parse_financial_report(tbname2, var2,freq='Q')
    BV[BV<=0]=np.nan #trick: delete samples with negative denominator
    roe=income/BV.shift(1,freq='3M') #trick:divide by one-quarter-lagged book equity
    #TODO: adjust with the announcement date
    '''
    It is a little different with the paper.To take time lag into consideration,we 
    just shift forward 6 month here but what the paper has done is " Earnings data
    in Compustat quarterly files are used in the months immediately after the most 
    recent public quarterly earnings announcement dates."
    '''
    roe=quaterly2monthly(roe, shift='6M')
    save(roe,'roe')

def get_bp():
    tbname='FI_T10'
    varname='F100401A' # 市净率
    bp=parse_financial_report(tbname,varname,consolidated=False)
    bp=quaterly2monthly(bp)
    save(bp,'bp')

def task(f):
    print(f)
    eval(f)()

if __name__=='__main__':
    fstrs=[f for f in locals().keys() if (f.startswith('get') and f not in
                                          ['get_ipython','get_inter_frame',
                                           'get_today'])]
    multiprocessing.Pool(6).map(task,fstrs)
    # for f in fstrs:#TODO:
    #     eval(f)()
    #     print(f)








