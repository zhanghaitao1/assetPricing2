# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-19  13:44
# NAME:assetPricing2-din.py

import pandas as pd
import numpy as np

from data.dataTools import read_df_from_gta, save, read_gta, read_unfiltered, load_data
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

def get_stockCloseD():
    #get daily stock close price
    tbname='TRD_Dalyr'
    varname='Clsprc'
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

def get_mktRetD():
    # get daily market return

    tbname = 'TRD_Cndalym'
    indVar = 'Trddt'
    targetVar = 'Cdretwdos'  # 考虑现金红利再投资的综合日市场回报率(流通市值加权平均法)
    df=read_gta(tbname)


    condition1=df['Markettype']==21 # 21=综合A股和创业板
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
    src=src[src['Nrr1']=='NRI01']
    src=src.set_index('Clsdt')

    rf=src[dic[freq]][2:]#delete the first two rows
    rf.index.name='t'
    rf.name='rf'+freq

    rf.index=pd.to_datetime(rf.index)
    if freq in ['W','M']:
        rf=rf.resample(freq).agg(lambda x:x[round(x.shape[0]/2)])

    return rf/100.0 #the unit of rf in the file is %,we adjust it to be actual value.

def get_rfD():
    '''
    get daily risk free return
    :return:
    '''
    df=_get_rf(freq='D')
    save(df,'rfD')

def get_rfM():
    '''
    get monthly risk free return

    :return:
    '''
    df=_get_rf(freq='M')
    save(df,'rfM')

def get_stockEretD():
    stockRetD=read_unfiltered('stockRetD')
    rfD=read_unfiltered('rfD')
    stockEretD=stockRetD.sub(rfD,axis=0)
    # The date for stockRetD is buisiness date,but for rfD, it is calendar date.
    stockEretD=stockEretD.dropna(axis=0,how='all')# use this to ajust the index from calendar date to buisiness date

    save(stockEretD,'stockEretD')

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

def get_stockCloseM():
    '''
    monthly stock close price
    :return:
    '''
    tbname = 'TRD_Mnth'
    varname='Mclsprc'
    indname='Trdmnt'
    colname='Stkcd'

    df=read_df_from_gta(tbname, varname, indname, colname)

    #TODO: identify the axis and convert the axis automatically
    df.index.name='t'
    df.columns.name='sid'
    df.index=freq_end(df.index, 'M')
    df.columns=df.columns.astype(str)

    save(df, 'stockCloseM')

def get_stockEretM():
    stockRetM=read_unfiltered('stockRetM')
    rfM=read_unfiltered('rfM')
    stockEretM=stockRetM.sub(rfM,axis=0)
    save(stockEretM,'stockEretM')

def get_mktRetM():
    tbname = 'TRD_Cnmont'
    indVar = 'Trdmnt'
    targetVar = 'Cmretwdos'  # 考虑现金红利再投资的综合日市场回报率(流通市值加权平均法)

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
    varname='Msmvosd' #月个股流通市值，单位 千元 #TODO:the unit convert it to million as Cakici, Chan, and Topyan, “Cross-Sectional Stock Return Predictability in China.”
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
    varname = 'F091001A'
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

#get stock close price yearly
def get_stockCloseY():
    tbname='TRD_Year'
    varname='Yclsprc'
    indname='Trdynt'
    colname='Stkcd'
    df=read_df_from_gta(tbname, varname, indname, colname)
    df.index=freq_end(df.index,'Y')
    df.index.name='t'
    df.columns=df.columns.astype(str)
    df.columns.name='sid'

    save(df,'stockCloseY')

def get_rpM():
    rpM=read_unfiltered('ff3M')['rp']
    rpM.name='rpM'
    save(rpM,'rpM')

def get_rpD():
    rpD=read_unfiltered('ff3D')['rp']
    rpD.name='rpD'
    save(rpD,'rpD')

def get_capmM():
    rpM=load_data('rpM')
    rpM.name='rp'
    save(rpM,'capmM',axis_info=False)

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

def get_ff3M():
    df=read_gta('STK_MKT_ThrfacMonth')
    #P9709 全部A股市场包含沪深A股和创业板
    #流通市值加权
    df=df[df['MarkettypeID']=='P9709'][['TradingMonth','RiskPremium1','SMB1','HML1']]
    df.columns=['t','rp','smb','hml']
    df=df.set_index('t')
    df.index=freq_end(df.index,'M')
    df.columns.name='type'
    save(df,'ff3M')

def get_ffcM():
    df=read_gta('STK_MKT_CarhartFourFactors')
    # P9709 全部A股市场包含沪深A股和创业板
    # 流通市值加权
    df = df[df['MarkettypeID'] == 'P9709'][
        ['TradingMonth', 'RiskPremium1', 'SMB1', 'HML1', 'UMD2']]
    df.columns = ['t', 'rp', 'smb', 'hml', 'mom']
    df.columns.name='type'
    df = df.set_index('t')
    df.index=freq_end(df.index,'M')

    save(df,'ffcM')

def get_ff5M():
    df=read_gta('STK_MKT_FivefacMonth')
    #P9709 全部A股市场包含沪深A股和创业板
    #流通市值加权
    #2*3 投资组合
    df=df[(df['MarkettypeID']=='P9709') & (df['Portfolios']==1)][
        ['TradingMonth','RiskPremium1','SMB1','HML1','RMW1','CMA1']]
    df.columns=['t','rp','smb','hml','rmw','cma']
    df.columns.name='type'
    df=df.set_index('t')
    df.index=freq_end(df.index,'M')
    # df.index.name='t'

    save(df,'ff5M')

def get_hxz4M():
    '''
    D:\app\python27\zht\researchTopics\assetPricing\calFactors.py\get_hxz4Factors()

    :return:
    '''

    fp=r'D:\zht\database\quantDb\researchTopics\assetPricing\benchmarkModel\hxz4.csv'
    df=pd.read_csv(fp,index_col=0)
    df.index=freq_end(df.index,'M')
    df.index.name='t'
    df.columns.name='type'
    save(df,'hxz4M')

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

def get_listInfo():
    df=read_gta('IPO_Cobasic', encoding='gbk')
    df=df.set_index('Stkcd')
    df.index.name='sid'
    df.index=df.index.astype(str)
    df.columns.name='type'
    #TODO: refer to page12 of 动量因子_164.pdf   ' 1 代表剔除金融、保险、 ST 类股票'
    df['not_financial']=df['Indcd']!=1 #financial stocks
    df['not_cross']=~df['Crcd'].notnull()#crosss means stocks listed on multiple stock markets
    df['is_sh']=df['Listexg']==1#listed on shanghai
    df['is_sz']=df['Listexg']==2#listed on shenzhen
    # Listdt denotes listed date ,'Ipodt' denotes IPO date
    df['Listdt']=df['Listdt'].replace(['0000-00-00','2100-01-01'],np.nan) #there is some invalid data in column 'Listdt
    df['listDate']=pd.to_datetime(df['Listdt'])
    df=df[['listDate','not_financial','not_cross','is_sh','is_sz']]
    df=df[~df.index.duplicated(False)] #there are some duplicated items such as '600018
    df=df.dropna()
    save(df,'listInfo')

def get_stInfo():
    '''
    for freq='M',delete all the months as long as ST or *ST appear in any day of that month,
    for freq='D',we only delete the current day with ST or *ST
    :return:
    '''
    #TODO: how about PT
    df=read_gta('TRD_Dalyr', encoding='gbk')
    df=df[['Trddt','Stkcd','Trdsta']]
    df.columns=['t','sid','status']
    df.columns.name='type'
    df['t']=pd.to_datetime(df['t'])
    df['sid']=df['sid'].astype(str)

    # df0=df[~df['status'].isin([2.0,3.0])] #TODO:
    df0=df[df['status']==1.0].copy() #TODO:or just delete all the ST stocks rather than just delete the corresponding months or days
    df0['not_st']=True
    dfD=df0.set_index(['t','sid'])['not_st']
    dfD=dfD.sort_index(level='t')
    dfD=dfD.unstack()

    def func(x):
        # for information about the status refer to the documents
        result= (2.0 not in x['status'].values) & (3.0 not in x['status'].values)
        return result

    #delete st months for each stock
    df1=df.groupby([pd.Grouper(key='t',freq='M'),'sid']).filter(func)
    dfM=df1.groupby([pd.Grouper(key='t',freq='M'),'sid']).sum()
    dfM['not_st']=True
    dfM=dfM['not_st']
    dfM=dfM.unstack()

    save(dfD,'stInfoD')
    save(dfM,'stInfoM')

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

def get_op():
    # --------------operating probability---------------
    tbname='FS_Comins'
    # var1='B001101000' # 营业收入
    #
    # var2='B001201000' # 营业成本
    # var3='B001209000' # 销售费用
    # var4='B001210000' # 管理费用
    # var5='B001211000' # 财务费用

    var6='B001300000' # 营业利润
    # var7='Bbd1102203' # 利息支出

    df=read_gta(tbname)
    df=df[df['Typrep']=='A'] #合并报表
    # df['op1']=df[var1]-df[var2]-df[var3]-df[var4]-df[var5]
    # df['op2']=df[var6]-df[var7]
    df['op']=df[var6]

    varname='op'
    indname='Accper'
    colname='Stkcd'
    OP=pd.pivot_table(df, varname, indname, colname)
    OP.index.name= 't'
    OP.index=pd.to_datetime(OP.index)
    OP.columns=OP.columns.astype(str)
    OP.columns.name= 'sid'

    #----------------book value---------------
    tbname='FS_Combas'
    #book value
    var1='A003000000' # 所有者权益合计
    var2='A003100000' # 归属于母公司所有者权益合计
    df=read_gta(tbname)
    df=df[df['Typrep']=='A'] # 合并报表
    df['bv1']=df[var1]
    df['bv']=df[var2]

    varname='bv'
    indname='Accper'
    colname='Stkcd'
    BV=pd.pivot_table(df, varname, indname, colname)
    BV.index.name= 't'
    BV.index=pd.to_datetime(BV.index)
    BV.columns=BV.columns.astype(str)
    BV.columns.name= 'sid'

    OP,BV=get_inter_frame([OP,BV])

    op=OP/BV
    op=op[op.index.month==12]
    op=op.shift(1,freq='6M')
    newIndex=pd.date_range(op.index[0],op.index[-1],freq='M')
    op=op.reindex(index=newIndex)
    op=op.fillna(method='ffill',limit=11)
    op.index.name='t'
    op.columns.name='sid'
    save(op,'op')

def get_inv():
    tbname='FS_Combas'
    #book value
    var='A001000000' # 总资产
    df=read_gta(tbname)
    df=df[df['Typrep']=='A'] # 合并报表
    df['ta']=df[var]

    varname = 'ta'
    indname = 'Accper'
    colname = 'Stkcd'
    ta = pd.pivot_table(df, varname, indname, colname)
    ta.index.name = 't'
    ta.index = pd.to_datetime(ta.index)
    ta.columns = ta.columns.astype(str)
    ta.columns.name = 'sid'
    ta=ta[ta.index.month==12]
    inv=ta.pct_change()
    inv=inv.shift(1,freq='6M')

    newIndex = pd.date_range(inv.index[0], inv.index[-1], freq='M')
    inv = inv.reindex(index=newIndex)
    inv = inv.fillna(method='ffill', limit=11)
    inv.index.name = 't'
    inv.columns.name = 'sid'

    save(inv, 'inv')


# if __name__=='__main__':
#     fstrs=[f for f in locals().keys() if (f.startswith('get') and f!='get_ipython')]
#     for f in fstrs:#TODO:
#         eval(f)()
#         print(f)








