#-*-coding: utf-8 -*-
#author:tyhj
#getBaseData.py 2017/7/29 9:59

import numpy as np
import pandas as pd
import os
import statsmodels.api as sm

from zht.util.dfFilter import filterDf
from zht.researchTopics.crossSection.params import sp
from zht.researchTopics.crossSection.dataAPI import get_df,save_df
from zht.util import mathu
from zht.util import pandasu


#涉及report的表
def _get_indictor1(name,tbname,fldname,timefld='Accper'):
    df=pd.read_csv(os.path.join(sp,tbname+'.csv'))
    df=df[df['Typrep']=='A']
    q='Accper endswith 12-31'
    df=filterDf(df,q)
    colnames=['Stkcd',timefld,fldname]
    df=df[colnames]
    subdfs=[]
    for stockId,x in list(df.groupby('Stkcd')):
        tmpdf=x[[timefld,fldname]]
        tmpdf=tmpdf.set_index(timefld)
        tmpdf.index=[ind[:-3] for ind in tmpdf.index]
        tmpdf.columns=[stockId]
        subdfs.append(tmpdf)

    table=pd.concat(subdfs,axis=1)
    table=table.sort_index(ascending=True)
    save_df(table,name)

def get_items1():
    items=[('operIncom','FS_Comins','B001101000','Accper'),
            ('operExp','FS_Comins','B001209000','Accper'),
            ('admExp','FS_Comins','B001210000','Accper'),
            ('finanExp','FS_Comins','B001211000','Accper'),
            ('totAsset','FS_Combas','A001000000','Accper'),
            ('totShe','FS_Combas','A003000000','Accper')
           ]
    for item in items:
        name,tbname,fldname,timefld=item
        _get_indictor1(name,tbname,fldname,timefld)
        print name

def get_op():
    operIncom=get_df('operIncom')
    admExp=get_df('admExp').fillna(method='ffill')
    finanExp=get_df('finanExp').fillna(method='ffill')
    operExp=get_df('operExp').fillna(method='ffill')

    op=operIncom-admExp-finanExp-operExp
    op.index=[str(int(ind[:4])+1)+'-06' for ind in op.index] #TODO:In june of year t,op comes from fiscal year ending in year t-1
    save_df(op,'op')

def get_inv():
    totAsset=get_df('totAsset')
    inv=totAsset.pct_change()
    inv.index = [str(int(ind[:4]) + 1) + '-06' for ind in inv.index]#TODO:In june of year t,inv comes from fiscal year ending in year t-1
    save_df(inv,'inv')

#涉及 trading 数据的表
def _get_indictor2(name,tbname,fldname,timefld='Trdmnt'):
    df = pd.read_csv(os.path.join(sp, tbname + '.csv'))
    q='Markettype in [1,4,16]'
    df=filterDf(df,q)
    colnames=['Stkcd',timefld,fldname]

    df = df[colnames]
    subdfs = []
    for stockId, x in list(df.groupby('Stkcd')):
        tmpdf = x[[timefld, fldname]]
        tmpdf = tmpdf.set_index(timefld)
        tmpdf.columns = [stockId]
        subdfs.append(tmpdf)

    table = pd.concat(subdfs, axis=1)
    table = table.sort_index(ascending=True)
    save_df(table, name)

def get_items2():
    items=[
        ('ret','TRD_Mnth','Mretwd','Trdmnt'),
        ('size','TRD_Mnth','Msmvosd','Trdmnt'),

    ]
    for item in items:
        name,tbname,fldname,timefld=item
        _get_indictor2(name,tbname,fldname,timefld)
        print name

def get_weight():
    size=get_df('size')
    weight=size.shift(1)
    save_df(weight,'weight')

def get_rf():
    df = pd.read_csv(os.path.join(sp,'TRD_Nrrate.csv'))
    q = 'Nrr1 == NRI01'  # TODO:TBC=国债票面利率
    df = filterDf(df, q)
    colnames = ['Clsdt', 'Nrrmtdt']
    df = df.sort_values('Clsdt')

    df = df[colnames]
    df = df.set_index('Clsdt')

    dates = pd.date_range(df.index[0], df.index[-1], freq='D')
    dates = [d.strftime('%Y-%m-%d') for d in dates]

    newdf = pd.DataFrame(index=dates)
    newdf['Nrrmtdt'] = df['Nrrmtdt']
    newdf = newdf.fillna(method='ffill')
    newdf = newdf.reset_index()
    newdf['month'] = newdf['index'].apply(lambda x: '-'.join(x.split('-')[:-1]))

    avg = newdf.groupby('month').mean()
    avg = avg / 100
    del avg.index.name
    avg.columns = ['rf']
    save_df(avg,'rf')

def get_rm1():
    Mretwd=get_df('ret')
    weight=get_df('weight')

    Mretwd.head()
    rm=pd.DataFrame()

    for month in Mretwd.index.tolist():
        tmpdf=pd.DataFrame(columns=Mretwd.columns)
        tmpdf.loc['ret']=Mretwd.loc[month]
        tmpdf.loc['weight']=weight.loc[month]
        tmpdf=tmpdf.T
        tmpdf=tmpdf.dropna(axis=0,how='any')
        if tmpdf.shape[0]>0:
            ret=np.average(tmpdf['ret'],weights=tmpdf['weight'])
            rm.loc[month,'rm']=ret
        else:
            rm.loc[month,'rm']=np.NaN
        print month

    save_df(rm,'rm1')

def get_rm():
    name='rm'
    dbname=''
    tbname='TRD_Cnmont'
    fldname='Cmretwdos'
    timefld='Trdmnt'
    q=[]
    cols=[]

    df = pd.read_csv(os.path.join(sp, tbname + '.csv'))
    q = 'Markettype == 5'#综合A股市场
    df = filterDf(df, q)
    colnames = [timefld, fldname]

    df = df[colnames]

    df=df.set_index('Trdmnt')
    df=df.sort_index()
    del df.index.name
    df.columns=['rm']
    save_df(df,'rm')

def get_rp():
    rf=get_df('rf')
    rm=get_df('rm')

    rp=rm['rm']-rf['rf']

    rp=rp.to_frame()
    rp.columns=['rp']
    save_df(rp,'rp')

def get_mv():
    name='mv'
    tbname='TRD_Mnth'
    fldname='Mclsprc' #月收盘价
    timefld='Trdmnt'

    df = pd.read_csv(os.path.join(sp, tbname + '.csv'))
    q1 = 'Markettype in [1,4,16]'
    q2= 'Trdmnt endswith 12' #TODO: only need the data in December
    q=[q1,q2]
    df = filterDf(df, q)
    colnames = ['Stkcd', timefld, fldname]

    df = df[colnames]
    subdfs = []
    for stockId, x in list(df.groupby('Stkcd')):
        tmpdf = x[[timefld, fldname]]
        tmpdf = tmpdf.set_index(timefld)
        tmpdf.columns = [stockId]
        subdfs.append(tmpdf)

    table = pd.concat(subdfs, axis=1)
    table = table.sort_index(ascending=True)
    save_df(table, name)

def get_bv():
    name='bv'
    tbname='FI_T9'
    fldname='F091001A' #每股净资产
    timefld='Accper'

    df = pd.read_csv(os.path.join(sp, tbname + '.csv'))

    q1 = 'Typrep == A'
    q2 = 'Accper endswith 12-31'  # TODO: only need annual report
    q = [q1, q2]
    df = filterDf(df, q)
    colnames = ['Stkcd', timefld, fldname]

    df = df[colnames]
    subdfs = []
    for stockId, x in list(df.groupby('Stkcd')):
        tmpdf = x[[timefld, fldname]]
        tmpdf = tmpdf.set_index(timefld)
        tmpdf.columns = [stockId]
        subdfs.append(tmpdf)

    table = pd.concat(subdfs, axis=1)
    table.index=[ind[:-3] for ind in table.index]
    table = table.sort_index(ascending=True)

    save_df(table, name)

def get_btm1():
    bv=get_df('bv')
    mv=get_df('mv')

    inds=sorted(list(set(bv.index.tolist()).intersection(set(mv.index.tolist()))))
    cols=sorted(list(set(bv.columns.tolist()).intersection(set(mv.columns.tolist()))))

    bv=bv.loc[inds,cols]
    mv=mv.loc[inds,cols]

    btm=bv/mv

    btm.index = [str(int(ind[:4]) + 1) + '-06' for ind in btm.index]
    #TODO: notice that we can only use the data 6 months later
    save_df(btm,'btm')

def get_btm2():
    size=get_df('size') #TODO:流通市值对应所有者权益？，总市值？
    totshe=get_df('totshe')
    size,totshe=pandasu.get_inter_frame(size,totshe)
    btm=totshe/size
    btm.index = [str(int(ind[:4]) + 1) + '-06' for ind in btm.index]
    save_df(btm,'btm2')

def get_nfc():
    '''
    non-financial stock codes
    :return:
    '''
    tbname='TRD_Co'
    df=pd.read_csv(os.path.join(sp,tbname+'.csv'))
    nf=df[df['Indcd']!=1]
    nf=nf.reset_index()
    nf=nf[['Stkcd']]
    save_df(nf,'nfc')

def get_factorId(tbname,param):
    '''
    :param tbname:
    :param param:an int number or a list of breakpoints such as [0.3,0.7]
    :return:
    '''
    df=get_df(tbname)
    nfc=get_df('nfc')['Stkcd'].values
    nfc=[str(n) for n in nfc]
    cols=[s for s in df.columns if s in nfc]
    df=df[cols]

    ids=pd.DataFrame(columns=df.columns)
    for month in df.index.tolist():
        sub = df.loc[month].to_frame()
        sub = sub.dropna()
        sub['id'] = mathu.getPortId(sub, param)
        ids.loc[month] = sub['id']
        print month
    save_df(ids, '%sId_%s' %(tbname,param))

def get_ids():
    get_factorId('size',2)
    get_factorId('btm',[0.3,0.7])

    get_factorId('size',5)
    get_factorId('btm',5)

def get_portId(sizeNum,btmNum):
    sizeId=get_df('sizeId_%s'%sizeNum)
    if btmNum==3:
        btmId=get_df('btmId_[0.3, 0.7]') #TODO: change the file name
    else:
        btmId=get_df('btmId_%s'%btmNum)

    sizeId,btmId=pandasu.get_inter_frame(sizeId,btmId)

    portId=sizeId*10+btmId
    save_df(portId,'portId_%s_%s'%(sizeNum,btmNum))

# get_portId(2,3)
# get_portId(5,5)

def portDescribe():
    portId=get_df('portId_2_3')
    size=pd.DataFrame(columns=[11,12,13,21,22,23])
    for month in portId.index.tolist():
        tmp=portId.loc[month]
        size.loc[month]=tmp.value_counts()
    size=size.fillna(0)
    describe=size.copy()
    describe['min']=size.min(axis=1)
    describe['max']=size.max(axis=1)
    describe['sum']=size.sum(axis=1)

    describe.to_csv(r'D:\quantDb\researchTopics\crossSection\data\observe\portDescribe.csv')

def get_smb_hml():
    portId=get_df('portId_2_3')
    portId=portId.T #TODO: transpose
    ret=get_df('ret')
    weight=get_df('weight')


    def _get_portRet(validmonth,stocks):
        tmp = pd.DataFrame()
        if validmonth in ret.index.tolist():
            tmp['ret'] = ret.loc[validmonth, stocks]  # TODO: validmonth rathar than month
            tmp['weight'] = weight.loc[validmonth, stocks]
            tmp = tmp.dropna(axis=0, how='any')
            portRet = pandasu.mean_self(tmp, 'ret', 'weight')
            return portRet
        else:
            return np.NaN

    factor = pd.DataFrame(columns=['smb', 'hml'])
    months = portId.columns.tolist()
    for month in months:
        year=month[:4]
        validmonths=[year+'-0'+str(i) for i in range(7,10)]
        validmonths+=[year+'-1'+str(i) for i in range(3)]
        validmonths+=[str(int(year)+1)+'-0'+str(i) for i in range(1,7)]

        aa = portId[portId[month] == 11].index.tolist()
        ab = portId[portId[month] == 12].index.tolist()
        ac = portId[portId[month] == 13].index.tolist()

        ba = portId[portId[month] == 21].index.tolist()
        bb = portId[portId[month] == 22].index.tolist()
        bc = portId[portId[month] == 23].index.tolist()

        for validmonth in validmonths:
            aaret=_get_portRet(validmonth,aa)
            abret=_get_portRet(validmonth,ab)
            acret=_get_portRet(validmonth,ac)

            baret=_get_portRet(validmonth,ba)
            bbret=_get_portRet(validmonth,bb)
            bcret=_get_portRet(validmonth,bc)

            smb=(aaret+abret+acret)/3.0-(baret+bbret+bcret)/3.0
            hml=(acret+bcret)/2.0-(aaret+baret)/2.0
            factor.loc[validmonth]=[smb,hml]
            print validmonth
    save_df(factor,'smb_hml')

def get_GTAff3():
    ff3 = pd.read_csv(r'D:\quantDb\sourceData\gta\data\csv\STK_MKT_ThrfacMonth.csv')
    ff3=ff3[ff3['MarkettypeID']=='P9709'] #综合A股和创业板
    ff3=ff3.set_index('TradingMonth')
    del ff3['MarkettypeID']
    del ff3.index.name
    save_df(ff3,'ff3')

def get_portRet():
    portId=get_df('portId_5_5')
    portId=portId.T

    ret=get_df('ret')
    weight=get_df('weight')

    ports=np.sort([p for p in portId.iloc[:,-1].unique() if not np.isnan(p)])

    portRet=pd.DataFrame()
    for month in portId.columns.tolist():
        year=month[:4]
        validmonths = [year + '-0' + str(i) for i in range(7, 10)]
        validmonths += [year + '-1' + str(i) for i in range(3)]
        validmonths += [str(int(year) + 1) + '-0' + str(i) for i in range(1, 7)]

        for port in ports:
            stocks=portId[portId[month]==port].index.tolist()
            for validmonth in validmonths:
                if validmonth in ret.index.tolist():
                    tmp=pd.DataFrame()
                    tmp['ret']=ret.loc[validmonth,stocks]
                    tmp['weight']=weight.loc[validmonth,stocks]
                    tmp=tmp.dropna(axis=0,how='any')
                    pr=pandasu.mean_self(tmp,'ret','weight')
                    portRet.loc[validmonth,port]=pr
                else:
                    portRet.loc[validmonth,port]=np.NaN
        print month
    portRet=portRet.dropna(axis=0,how='any')
    save_df(portRet,'portRet')

def get_portEret():
    portRet=get_df('portRet')
    rf=get_df('rf')
    portEret=portRet.sub(rf['rf'],axis=0)
    portEret=portEret.dropna(axis=0,how='any')
    save_df(portEret,'portEret')

def get_portRet_ts():
    port=pd.read_csv(r'D:\quantDb\resset\PMONRET_FF.csv')
    q1='Exchflg == 0'
    q2='Mktflg == A'
    port=filterDf(port,[q1,q2])
    port['Date']=[d[:-3] for d in port['Date']]

    months=sorted(port['Date'].unique().tolist())

    portRet_rs_tmv=pd.DataFrame()
    portRet_rs_mc=pd.DataFrame()
    for month in months:
        for i in range(1,6):
            for j in range(1,6):
                try:
                    portRet_rs_tmv.loc[month,i*10+j]=port[(port['Date']==month) & (port['Sizeflg']==i)
                                            & (port['BMflg']==j)]['Pmonret_tmv'].values[0]
                except IndexError:
                    portRet_rs_tmv.loc[month,i*10+j]=np.NaN
                try:
                    portRet_rs_mc.loc[month,i*10+j]=port[(port['Date']==month) & (port['Sizeflg']==i)
                                            & (port['BMflg']==j)]['Pmonret_mc'].values[0]
                except IndexError:
                    portRet_rs_mc.loc[month,i*10+j]=np.NaN
        print month

    save_df(portRet_rs_tmv,'portRet_rs_tmv')
    save_df(portRet_rs_mc,'portRet_rs_mc')

def get_portEret_ts():
    portRet_rs_tmv=get_df('portRet_rs_tmv')
    portRet_rs_mc=get_df('portRet_rs_mc')
    rf=get_df('rf')

    portEret_rs_tmv = portRet_rs_tmv.sub(rf['rf'], axis=0)
    portEret_rs_mc = portRet_rs_mc.dropna(axis=0, how='any')
    save_df(portEret_rs_tmv, 'portEret_rs_tmv')
    save_df(portEret_rs_mc, 'portEret_rs_mc')

#===========================================validate=============================================
def validate_rp():
    myrp=get_df('rp')
    myrp.columns=['myrp']

    rp=pd.read_csv(os.path.join(sp,'STK_MKT_FivefacMonth.csv'))
    rp = rp[rp['Portfolios'] == 1]

    rp = rp[rp['MarkettypeID'] == 'P9709']

    colnames = ['TradingMonth', 'RiskPremium1', 'RiskPremium2']
    rp = rp[colnames]
    rp = rp.set_index('TradingMonth')

    com=pd.concat([myrp,rp],axis=1)
    com=com.dropna(axis=0,how='any')

    ax=com.cumsum().plot()
    fig=ax.get_figure()
    fig.savefig(r'D:\quantDb\researchTopics\crossSection\data\observe\rp.png')

def validate_smb_hml():
    smb_hml=get_df('smb_hml')

    ff3=get_df('ff3')

    com=pd.concat([smb_hml,ff3],axis=1)
    com=com.dropna(axis=0,how='any')
    cols1=['smb','SMB1','SMB2']
    cols2=['hml','HML1','HML2']

    com[cols1][24:].cumsum().plot().get_figure().savefig(r'D:\quantDb\researchTopics\crossSection\data\observe\smb.png')
    com[cols2][24:].cumsum().plot().get_figure().savefig(r'D:\quantDb\researchTopics\crossSection\data\observe\hml.png')

def validate_portRet():
    df=pd.read_csv(r'D:\quantDb\resset\PMONRET_FF.csv',index_col=0)
    q1='Exchflg == 0'
    q2='Mktflg == A'
    df=filterDf(df,[q1,q2])

    mypr=get_df('portRet')
    mypr.columns=[int(float(col)) for col in mypr.columns]
    pr=pd.DataFrame()
    for date in sorted(df['Date'].unique().tolist()):
        month=date[:-3]
        for i in range(1,6):
            for j in range(1,6):
                try:
                    pr.loc[month,i*10+j]=df[(df['Sizeflg']==i) & (df['BMflg']==j) & (df['Date']==date)]['Pmonret_tmv'].values[0]
                except IndexError:
                    pr.loc[month,i*10+j]=np.NaN
                    pass

        print month

    pr=pr.dropna(axis=0,how='any')

    for i in range(1,6):
        for j in range(1,6):
            port=i*10+j
            tmp=pd.DataFrame()
            tmp['mypr']=mypr[port]
            tmp['pr']=pr[port]
            tmp.cumsum().plot().get_figure().savefig(r'D:\quantDb\researchTopics\crossSection\data\observe\portRet\%s.png'%port)

def validate_rm():
    rm = get_df('rm')
    rm1 = get_df('rm1')

    df = pd.DataFrame()
    df['rm'] = rm['rm']
    df['rm1'] = rm1['rm']
    df.cumsum().plot()

    (df['rm'] - df['rm1']).cumsum().plot()

def validate_bv():
    name = 'bv1'
    tbname = 'FAR_Finidx'
    fldname = 'T60300'
    timefld = 'Accper'

    df = pd.read_csv(os.path.join(sp, tbname + '.csv'))
    colnames = ['Stkcd', timefld, fldname]
    df = df[colnames]
    subdfs = []
    for stockId, x in list(df.groupby('Stkcd')):
        tmpdf = x[[timefld, fldname]]
        tmpdf = tmpdf.set_index(timefld)
        tmpdf.index = [ind[:-3] for ind in tmpdf.index]
        tmpdf.columns = [stockId]
        subdfs.append(tmpdf)

    table = pd.concat(subdfs, axis=1)
    table = table.sort_index(ascending=True)
    save_df(table, name)

#===================================================================================================

#=====================================================================================================




