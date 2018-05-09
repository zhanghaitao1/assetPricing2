# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-09  16:23
# NAME:assetPricing2-Chen2010.py
from core.ff5 import regression_details_5x5
from data.dataApi import Database
from data.dataTools import read_unfiltered
from data.din import parse_financial_report, yearly2monthly
from tool import assign_port_id, my_average
from zht.data.gta.api import read_gta
import os
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
import numpy as np



direc= r'D:\zht\database\quantDb\researchTopics\assetPricing2\data\FI'
dirData= r'D:\zht\database\quantDb\researchTopics\assetPricing2\data\FI\data'
dirFig=r'D:\zht\database\quantDb\researchTopics\assetPricing2\data\FI\fig'
dirFactor=r'D:\zht\database\quantDb\researchTopics\assetPricing2\data\FI\factor'

def _save(df,name):
    df.to_pickle(os.path.join(dirData, name + '.pkl'))

def _read(name):
    return pd.read_pickle(os.path.join(dirData, name + '.pkl'))


#parse all the financial indicators
def _filter_indicators(lst):
    newlst=[]
    mark=[]
    for ele in lst:
        if ele[-1].isdigit():
            newlst.append(ele)
        else:
            if ele[:-1] not in mark:
                newlst.append(ele)
                mark.append(ele[:-1])
    return newlst

def parse_all_financial_indicators():
    tbnames=['FI_T{}'.format(i) for i in range(1,12)]
    for tbname in tbnames:
        df=read_gta(tbname)
        varnames=[col for col in df.columns if col not in
                  ['Accper','Indcd','Stkcd','Typrep']]

        if 'Typrep' in df.columns:
            consolidated=True
        else:
            consolidated=False

        varnames=_filter_indicators(varnames)
        for varname in varnames:
            df=parse_financial_report(tbname,varname,consolidated=consolidated)
            df=yearly2monthly(df)
            _save(df,'{}__{}'.format(tbname,varname))

            print(tbname,varname)



def indicator2factor(indicator):
    sampleControl=False
    q=5

    # data lagged
    df = _read(indicator)
    s = df.stack()
    s.name = indicator
    weight = Database(sample_control=sampleControl).by_indicators(['weight'])
    datalagged = pd.concat([s, weight], axis=1)
    datalagged = datalagged.groupby('sid').shift(1)

    # data t
    datat = Database(sample_control=sampleControl).by_indicators(['stockEretM'])
    comb = pd.concat([datalagged, datat], axis=1)
    comb = comb.dropna()

    comb['g'] = comb.groupby('t', group_keys=False).apply(
        lambda df: assign_port_id(df[indicator], q))

    panel = comb.groupby(['t', 'g']).apply(
        lambda df: my_average(df, 'stockEretM', wname='weight')) \
        .unstack(level=['g'])

    factor = panel[q] - panel[1]
    factor.name=indicator
    factor.to_pickle(os.path.join(dirFactor,'{}.pkl'.format(indicator)))

def _task(indicator):
    try:
        indicator2factor(indicator)
    except:
        with open(os.path.join(direc, 'failed.txt'), 'a') as f:
            f.write(indicator+'\n')
    print(indicator)

def multi_indicator2factor():
    indicators=[ind[:-4] for ind in os.listdir(dirData)]
    p=Pool(6)
    p.map(_task,indicators)



def analyse_corr():
    fns=os.listdir(dirFactor)

    ss=[pd.read_pickle(os.path.join(dirFactor,fn)) for fn in fns]
    comb=pd.concat(ss,axis=1)
    corr=comb.corr()

    cc=corr.corr()

    tri=corr.mask(np.triu(np.ones(corr.shape),k=0).astype(bool))

    tri=tri.stack().sort_values(ascending=False)

    tri=tri.reset_index()

    thresh=0.9

def plot_all():
    fns = os.listdir(dirFactor)

    ss = [pd.read_pickle(os.path.join(dirFactor, fn)) for fn in fns]
    comb = pd.concat(ss, axis=1)

    tup=[]
    for col,s in comb.items():
        sharpe_abs=abs(s.mean()/s.std())
        tup.append((col,sharpe_abs))

    tup=sorted(tup,key=lambda x:x[1],reverse=True)

    for i,ele in enumerate(tup):
        indicator=ele[0]
        s=comb[indicator]
        s=s.dropna()
        fig=plt.figure()
        plt.plot(s.index,s.cumsum())
        fig.savefig(os.path.join(dirFig,indicator+'.png'))
        print(i)

    sp=pd.DataFrame(tup,columns=['indicator','sharpe'])
    sp.to_pickle(os.path.join(direc, 'sharpe.pkl'))

sharpe=pd.read_pickle(os.path.join(direc, 'sharpe.pkl'))

indicator=sharpe['indicator'][0]
factor=pd.read_pickle(os.path.join(dirFactor,indicator+'.pkl'))


ff3=read_unfiltered('ff3M')
model=pd.concat([ff3[['rp','smb']],factor],axis=1)
model=model.dropna()

tableas=[]
tablets=[]
for bench in [ff3,model]:
    tablea,tablet=regression_details_5x5(bench)
    tableas.append(tablea)
    tablets.append(tablet)

comba=pd.concat(tableas,axis=0,keys=['ff3','myModel'])
combt=pd.concat(tablets,axis=0,keys=['ff3','myModel'])

comba.to_csv(os.path.join(direc,'comba.csv'))
combt.to_csv(os.path.join(direc,'combt.csv'))


#TODO: wrong ,the factor should be neutralized by size




