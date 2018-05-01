# -*-coding: utf-8 -*-
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-20  19:36
# NAME:assetPricing2-din.py

'''
The standards for din.py:
1. all the data save as DataFrame even the series should be converted into DataFrame
2. The index of dataframe should be named
3. do not save the data with stacked multiIndex Dataframethe data unless you have to
4. for multiIndex dataframe,the first index should be named 't' and 'sid' for the second
5. notice the time index

'''



import pandas as pd
import os

from data.dataTools import save
from dout import read_df
from pandas.tseries.offsets import MonthEnd
from zht.data.gta.api import read_gta
from zht.data.resset.api import read_resset
from zht.data.wind.api import read_wind
from zht.utils.dateu import freq_end

from config import DATA_SRC, DATA_PATH
import numpy as np



def _readFromSrc(tbname):
    df=pd.read_csv(os.path.join(DATA_SRC,tbname+'.csv'))
    return df

def _get_df(tbname, varname, indname, colname, fn):
    '''
    get df by use pd.pivot_table

    :param tbname:table name
    :param varname:variable name in tbname
    :param indname:name in the table to be set as index of the returnd df
    :param colname:name in the table to be set as column of the returned df
    :param fn:the name of df to be saved
    :return:
    '''
    path=os.path.join(DATA_PATH, fn + '.csv')
    table=_readFromSrc(tbname)
    df=pd.pivot_table(table,varname,indname,colname)
    df.index=pd.to_datetime(df.index)
    df.to_csv(path)

def get_stockRetD():
    # get stock daily stock return
    tbname='TRD_Dalyr'
    varname='Dretwd'#考虑现金红利再投资的收益
    indname='Trddt'
    colname='Stkcd'
    fn='stockRetD'
    _get_df(tbname, varname, indname, colname, fn)

def get_mktRetD():
    # get daily market return
    newName = 'mktRetD'
    path=os.path.join(DATA_PATH, newName + '.csv')

    tbname = 'TRD_Cndalym'
    indVar = 'Trddt'

    targetVar = 'Cdretwdos'  # 考虑现金红利再投资的综合日市场回报率(流通市值加权平均法)
    df = _readFromSrc(tbname)
    condition1=df['Markettype']==21 # 21=综合A股和创业板
    df=df[condition1]

    df = df.set_index(indVar)
    df = df.sort_index()
    df = df[[targetVar]]
    del df.index.name
    df.columns = [newName]

    df.index = pd.to_datetime(df.index)
    df.to_csv(path)

def get_stockCloseD():
    #get daily stock close price
    tbname='TRD_Dalyr'
    varname='Clsprc'
    indname='Trddt'
    colname='Stkcd'
    fn='stockCloseD'
    _get_df(tbname, varname, indname, colname, fn)

def _get_rf(freq):
    '''
    parse risk free rate from the database
    Args:
        freq: D (daily),W (weekly),M (monthly)

    Returns:

    '''
    dic={'D':'Nrrdaydt','W':'Nrrwkdt','M':'Nrrmtdt'}

    tname = 'TRD_Nrrate'
    src = _readFromSrc(tname)
    #NRI01=定期-整存整取-一年利率；TBC=国债票面利率,根据复利计算方法，
    # 将年度的无风险利率转化为月度数据
    src=src[src['Nrr1']=='NRI01']
    src=src.set_index('Clsdt')
    del src.index.name

    rf=src[[dic[freq]]][2:]#delete the first two rows
    rf.columns=['rf'+freq]

    rf.index=pd.to_datetime(rf.index)
    if freq in ['W','M']:
        rf=rf.resample(freq).last()

    return rf/100.0 #the unit of rf in the file is %,we adjust it to be actual value.

def get_rfD():
    '''
    get daily risk free return
    :return:
    '''
    df=_get_rf(freq='D')
    df.to_csv(os.path.join(DATA_PATH, 'rfD.csv'))

def get_rfM():
    '''
    get monthly risk free return

    :return:
    '''
    df=_get_rf(freq='M')
    df.to_csv(os.path.join(DATA_PATH, 'rfM.csv'))

def get_eretD():
    stockRetD=read_df('stockRetD','D')
    rfD=read_df('rfD','D')
    eretD=stockRetD.sub(rfD['rfD'],axis=0)
    # The date for stockRetD is buisiness date,but for rfD, it is calendar date.
    eretD=eretD.dropna(axis=0,how='all')# use this to ajust the index from calendar date to buisiness date
    eretD.to_csv(os.path.join(DATA_PATH,'eretD.csv'))

def get_stockRetM():
    '''
    monthly stock return with dividend
    '''
    tbname = 'TRD_Mnth'
    varname='Mretwd'#考虑现金红利再投资的收益
    indname='Trdmnt'
    colname='Stkcd'
    fn='stockRetM'
    _get_df(tbname, varname, indname, colname, fn)

def get_stockCloseM():
    '''
    monthly stock close price
    :return:
    '''
    tbname = 'TRD_Mnth'
    varname='Mclsprc'#考虑现金红利再投资的收益
    indname='Trdmnt'
    colname='Stkcd'
    fn='stockCloseM'
    _get_df(tbname, varname, indname, colname, fn)

def get_eretM():
    stockRetM=read_df('stockRetM','M')
    rfM=read_df('rfM','M')
    eretM=stockRetM.sub(rfM['rfM'],axis=0)
    eretM.to_csv(os.path.join(DATA_PATH,'eretM.csv'))

def get_mktRetM():
    newName='mktRetM'
    path=os.path.join(DATA_PATH, newName + '.csv')
    tbname = 'TRD_Cnmont'
    indVar = 'Trdmnt'
    targetVar = 'Cmretwdos'  # 考虑现金红利再投资的综合日市场回报率(流通市值加权平均法)

    df = _readFromSrc(tbname)
    df=df[df['Markettype']==21]# 21=综合A股和创业板

    df = df.set_index(indVar)
    df = df.sort_index()
    df = df[[targetVar]]
    del df.index.name
    df.columns = [newName]
    df.index=freq_end(df.index, 'M')
    df.to_csv(path)

def get_capM():
    '''
    get stock monthly circulation market capitalization

    :return:
    '''
    tbname='TRD_Mnth'
    varname='Msmvosd' #月个股流通市值，单位 千元 #TODO:the unit convert it to million as Cakici, Chan, and Topyan, “Cross-Sectional Stock Return Predictability in China.”
    indname='Trdmnt'
    colname='Stkcd'
    fn='capM'
    _get_df(tbname, varname, indname, colname, fn)

# financial indicators-------------------------------------------
def get_bps_gta():
    tbname = 'FI_T9'
    varname = 'F091001A'
    indname = 'Accper'
    colname = 'Stkcd'
    fn = 'bps'
    _get_df(tbname, varname, indname, colname, fn)

def get_bps_wind():
    '''
    from code generator by use

    w.wsd("000001.SZ,000002.SZ,000004.SZ,000005.SZ,000006.SZ", "bps", "2017-02-04", "2018-03-05", "currencyType=;Period=Q;Fill=Previous")

    :return:
    '''
    df=read_wind('bps', freq='M')
    df.to_csv(os.path.join(DATA_PATH,'bps_wind.csv'))

#get stock close price yearly
def get_stockCloseY():
    tbname='TRD_Year'
    varname='Yclsprc'
    indname='Trdynt'
    colname='Stkcd'
    fn='stockCloseY'

    path=os.path.join(DATA_PATH, fn + '.csv')
    table=_readFromSrc(tbname)
    df=pd.pivot_table(table,varname,indname,colname)
    df.index=freq_end(df.index, 'Y')
    df.to_csv(path)

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
    df=df[['Rmrf_tmv','Smb_tmv','Hml_tmv']]#weighted with tradable capitalization
    df.columns=['rp','smb','hml']
    df.to_csv(os.path.join(DATA_PATH,'ff3M_resset.csv'))

def get_ff3M():
    df=read_gta('STK_MKT_ThrfacMonth')
    #P9709 全部A股市场包含沪深A股和创业板
    #流通市值加权
    df=df[df['MarkettypeID']=='P9709'][['TradingMonth','RiskPremium1','SMB1','HML1']]
    df.columns=['t','rp','smb','hml']
    df=df.set_index('t')
    df.to_csv(os.path.join(DATA_PATH,'ff3M.csv'))

def get_ffcM():
    df=read_gta('STK_MKT_CarhartFourFactors')
    # P9709 全部A股市场包含沪深A股和创业板
    # 流通市值加权
    df = df[df['MarkettypeID'] == 'P9709'][
        ['TradingMonth', 'RiskPremium1', 'SMB1', 'HML1', 'UMD2']]
    df.columns = ['t', 'rp', 'smb', 'hml', 'mom']
    df = df.set_index('t')
    df.to_csv(os.path.join(DATA_PATH, 'ffcM.csv'))

def get_ff5M():
    df=read_gta('STK_MKT_FivefacMonth')
    #P9709 全部A股市场包含沪深A股和创业板
    #流通市值加权
    #2*3 投资组合
    df=df[(df['MarkettypeID']=='P9709') & (df['Portfolios']==1)][
        ['TradingMonth','RiskPremium1','SMB1','HML1','RMW1','CMA1']]
    df.columns=['t','rp','smb','hml','rmw','cma']
    df=df.set_index('t')
    df.to_csv(os.path.join(DATA_PATH,'ff5M.csv'))


def get_hxz4M():
    '''
    D:\app\python27\zht\researchTopics\assetPricing\calFactors.py\get_hxz4Factors()

    :return:
    '''

    fp=r'D:\zht\database\quantDb\researchTopics\assetPricing\benchmarkModel\hxz4.csv'
    df=pd.read_csv(fp,index_col=0)
    df.index=pd.to_datetime(df.index)+MonthEnd(0)
    df.to_csv(os.path.join(DATA_PATH,'hxz4M.csv'))

def get_ff3D():
    tbname='STK_MKT_ThrfacDay'
    df=read_gta(tbname)
    condition1=df['MarkettypeID']=='P9707'
    # P9709 全部A股市场包含沪深A股和创业板.
    # 流通市值加权
    df = df[condition1][
        ['TradingDate', 'RiskPremium1', 'SMB1', 'HML1']]
    df.columns = ['t', 'rp', 'smb', 'hml']
    df = df.set_index('t')
    df.to_csv(os.path.join(DATA_PATH, 'ff3D.csv'))

def get_rpM():
    rpM=read_df('ff3M','M')[['rp']]
    rpM.to_csv(os.path.join(DATA_PATH,'rpM.csv'))

def get_capmM():
    '''
    just as get_rpM()
    :return:
    '''
    pass

def get_rpD():
    rpD=read_df('ff3D','D')[['rp']]
    rpD.to_csv(os.path.join(DATA_PATH,'rpD.csv'))

def get_listInfo():
    df=pd.read_csv(os.path.join(DATA_SRC,'IPO_Cobasic.csv'),encoding='gbk')
    df=df.set_index('Stkcd')
    df.index.name='sid'
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
    df.to_csv(os.path.join(DATA_PATH,'listInfo.csv'))

def get_listInfo1():
    fp=r'E:\a\gta20180412\txt\STK_ListedCoInfoAnl.txt'
    df = pd.DataFrame([row.split('\t') for row in open(fp, encoding='ISO-8859-1', newline='\r').readlines()])
    df=df[2:]

    #TODO: wrong!  find stock code online,the table only contains information from 2010.


def get_stInfo():
    '''
    for freq='M',delete all the months as long as ST or *ST appear in any day of that month,
    for freq='D',we only delete the current day with ST or *ST
    :return:
    '''
    #TODO: how about PT
    df=pd.read_csv(os.path.join(DATA_SRC,'TRD_Dalyr.csv'),encoding='gbk')
    df=df[['Trddt','Stkcd','Trdsta']]
    df.columns=['t','sid','status']
    df['t']=pd.to_datetime(df['t'])
    def func(df):
        # for information about the status refer to the documents
        result=(2.0 not in df['status'].values) & (3.0 not in df['status'].values)
        return result

    df1=df.groupby([pd.Grouper(key='t',freq='M'),'sid']).filter(func)

    dfM=df1.groupby([pd.Grouper(key='t',freq='M'),'sid']).sum()
    dfM['not_st']=True
    dfM=dfM['not_st']
    dfM=dfM.unstack()
    dfM.columns=dfM.columns.astype(str)


    df['not_st']=True
    dfD=df.set_index(['t','sid'])[['not_st']]
    dfD=dfD.sort_index(level='t')

    dfD=dfD['not_st'].unstack()
    dfD.columns=dfD.columns.astype(str)

    save(dfD,'stInfoD')
    save(dfM,'stInfoM')




# if __name__=='__main__':
#     fstrs=[f for f in locals().keys() if (f.startswith('get') and f!='get_ipython')]
#     for f in fstrs:
#         eval(f)()
#         print(f)


