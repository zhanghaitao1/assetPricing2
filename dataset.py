# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-22  15:15
# NAME:assetPricing2-dataset.py
from types import FunctionType

from config import DATA_PATH, DATA_UNI_PATH, TMP_PATH
from dout import read_df
import pandas as pd
import os
import pickle

'''
standard of the dataframe:
index.names=['t','sid']
columns=['type1','type2']
type1,or type2 can be rfM,eretM,and size or idioskewD_12M
'''

class Unify_indicators():
    def unify_beta(self):
        #beta
        betaD=pd.read_csv(os.path.join(DATA_PATH,'betaD.csv'),index_col=[0,1],parse_dates=True)
        betaM=pd.read_csv(os.path.join(DATA_PATH,'betaM.csv'),index_col=[0,1],parse_dates=True)
        #index: type_name,t
        #columns:sid
        betaD=betaD.stack()
        betaD.index.names=['type','t','sid']
        betaD=betaD.unstack('type')

        betaM=betaM.stack()
        betaM.index.names=['type','t','sid']
        betaM=betaM.unstack('type')

        comb=pd.concat([betaD,betaM],axis=1)
        return comb

    def unify_size(self):
        #size
        size=read_df('size','M')
        mktCap_ff=read_df('mktCap_ff','M')
        size_ff=read_df('size_ff','M')
        #index:t
        #columns:sid
        size=size.stack()
        size.name='size'
        mktCap_ff=mktCap_ff.stack()
        mktCap_ff.name='mktCap_ff'
        size_ff=size_ff.stack()
        size_ff.name='size_ff'
        comb=pd.concat([size,mktCap_ff,size_ff],axis=1)
        return comb

    def unify_value(self):
        #value
        bm=read_df('bm','M')
        logbm=read_df('logbm','M')
        bm=bm.stack()
        bm.index.names=['t','sid']
        logbm=logbm.stack()
        logbm.index.names=['t','sid']
        comb=pd.concat([bm,logbm],axis=1,keys=['bm','logbm'])
        return comb

    def unify_momentum(self):
        #mom
        momentum=pd.read_csv(os.path.join(DATA_PATH,'momentum.csv'),index_col=[0,1],parse_dates=True)
        #index:t,sid
        #columns:types
        return momentum

    def unify_reversal(self):
        #reversal
        reversal=pd.read_csv(os.path.join(DATA_PATH,'reversal.csv'),index_col=[0,1],parse_dates=True)
        #index:t,sid
        #columns:type
        return reversal

    def unify_liquidity(self):
        illiq=pd.read_csv(os.path.join(DATA_PATH,'illiq.csv'),index_col=[0,1],parse_dates=True)
        illiq=illiq.stack().unstack('type').head()
        illiq.index.names=['t','sid']

        liqBeta=read_df('liqBeta','M')
        liqBeta=liqBeta.stack()
        liqBeta.index.names=['t','sid']
        liqBeta.name='liqBeta'

        comb=pd.concat([illiq,liqBeta],axis=1).head()
        return comb

    def unify_skewness(self):
        #skewness
        dfs=[]
        for name in ['skewD','coskewD','idioskewD','skewM','coskewM','idioskewM']:
            df=pd.read_csv(os.path.join(DATA_PATH,name+'.csv'),index_col=[0,1],parse_dates=True)
            df=df.stack()
            df.index.names=['type','t','sid']
            df=df.unstack('type')
            df.columns=['_'.join([name,col]) for col in df.columns]
            dfs.append(df)
            print(name)
        comb=pd.concat(dfs,axis=1)
        return comb

    def unify_idiosyncraticVolatility(self):
        #idiosyncratic Volatility
        # index:types,t
        # columns:sid

        dfs=[]
        for name in ['volD','volssD','idioVol_capmD','idioVol_ff3D',
                     'volM','volssM','idioVol_capmM','idioVol_ff3M','idioVol_ffcM']:
            df=pd.read_csv(os.path.join(DATA_PATH,name+'.csv'),index_col=[0,1],parse_dates=True)
            df=df.stack()
            df.index.names=['type','t','sid']
            df=df.unstack('type')
            df.columns=['_'.join([name,col]) for col in df.columns]
            dfs.append(df)
            print(name)
        comb=pd.concat(dfs,axis=1)
        return comb

    def unify_eretM(self):
        #eretM
        eretM=read_df('eretM','M')
        eretM=eretM.stack().to_frame()
        eretM.columns=['eretM']
        return eretM

class Unify_base:
    def unify_rfM(self):
        rfM=read_df('rfM','M')
        return rfM

    def unify_mktRetM(self):
        mktRetM=read_df('mktRetM','M')
        return mktRetM

    def unify_rpM(self):
        rpM=read_df('rpM','M')
        return rpM

    def unify_capM(self):
        capM=read_df('capM','M')
        capM.index.name='t'
        return capM

    def unify_ff3M(self):
        ff3M=read_df('ff3M','M')
        return ff3M

    def unify_ffcM(self):
        ffcM=read_df('ffcM','M')
        return ffcM

    def unify_ff5M(self):
        ff5M=read_df('ff5M','M')
        return ff5M

    def unify_hxz4M(self):
        hxz4M=read_df('hxz4M','M')
        return hxz4M


class Dataset:
    def __init__(self):
        self.data,self.info=self.get_data_and_info()

    def _get_all_methods(self,cls):
        return [x for x, y in cls.__dict__.items() if type(y) == FunctionType]

    def _combine_all_indicators(self):
        methods1=self._get_all_methods(Unify_indicators)
        info={}
        _dfs=[]
        for m1 in methods1:
            df=getattr(Unify_indicators(),m1)()
            _dfs.append(df)
            info[m1.split('_')[1]]=df.columns.tolist()
        comb=pd.concat(_dfs,axis=1)
        return comb,info

    def _combine_base(self):
        #TODO: shift(1) for all the indicators?Refer to dataset.py in assetPricing1
        methods2=self._get_all_methods(Unify_base)
        _dfs2=[]
        for m2 in methods2:
            df=getattr(Unify_base(),m2)()
            _dfs2.append(df)
        df2=pd.concat(_dfs2,axis=1)
        #TODO:How to handle the duplicated index
        raise NotImplementedError

    def get_data_and_info(self):
        p_data=os.path.join(TMP_PATH,'tmp','data.obj')
        p_info=os.path.join(TMP_PATH,'tmp','info.obj')
        if os.path.isfile(p_data) and os.path.isfile(p_info):
            with open(p_data) as f:
                data=pickle.load(f)
            with open(p_info) as f:
                info=pickle.load(f)
            return data,info
        else:
            data,info=self._combine_all_indicators()
            pickle.dump(data,p_data)
            pickle.dump(info,p_info)
            return self._combine_all_indicators()

    @property
    def all_indicators(self):
        return sum(self.info.values(),[])

    def by_factor(self,factorname):
        return self.data[self.info[factorname]]

    def by_indicators(self,indicators):
        '''
        no mather indicators is just a string represent one indicators
        or list (tuple),the function will return a DataFrame
        :param indicators:
        :return: DataFrame
        '''
        if isinstance(indicators,(list,tuple)):
            return self.data[indicators]
        else:
            return self.data[[indicators]]

dataset=Dataset()



