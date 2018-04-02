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

from zht.utils.dfu import join_dfs

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
        comb.index.names=['t','sid']
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
        return momentum

    def unify_reversal(self):
        #reversal
        reversal=pd.read_csv(os.path.join(DATA_PATH,'reversal.csv'),index_col=[0,1],parse_dates=True)
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

def _add_prefix(df,prefix):
    oldCol=df.columns
    newCol=['__'.join([prefix,col]) for col in oldCol]
    df.columns=newCol
    return df

class Unify_base:
    #------------------------
    #multiIndex
    def unify_eretM(self):
        #eretM
        eretM=read_df('eretM','M')
        eretM=eretM.stack().to_frame()
        eretM.columns=['eretM']
        eretM.index.names=['t','sid']
        return eretM

    def unify_capM(self):
        '''
        market capitalization

        :return:
        '''
        capM=read_df('capM','M')
        capM=capM.stack().to_frame()
        capM.index.name='t'
        capM.columns=['capM']
        capM.index.names=['t','sid']
        return capM

    #------------------------------
    #single index
    def unify_rfM(self):
        rfM=read_df('rfM','M')
        return rfM

    def unify_mktRetM(self):
        mktRetM=read_df('mktRetM','M')
        return mktRetM

    def unify_rpM(self):
        rpM=read_df('rpM','M')
        return rpM

    #--------------------------
    #benchmark model with single index
    def unify_ff3M(self):
        ff3M=read_df('ff3M','M')
        ff3M=_add_prefix(ff3M,'ff3M')
        return ff3M

    def unify_ffcM(self):
        ffcM=read_df('ffcM','M')
        ffcM=_add_prefix(ffcM,'ffcM')
        return ffcM

    def unify_ff5M(self):
        ff5M=read_df('ff5M','M')
        ff5M=_add_prefix(ff5M,'ff5M')
        return ff5M

    def unify_hxz4M(self):
        hxz4M=read_df('hxz4M','M')
        hxz4M=_add_prefix(hxz4M,'hxz4M')
        return hxz4M

class Base:
    def __init__(self,cls):
        self.cls=cls
        self.name=cls.__name__
        self.data,self.info=self._combine_all_indicators()

    def _get_all_methods(self):
        return [x for x, y in self.cls.__dict__.items() if type(y) == FunctionType]

    def _combine_all_indicators(self):
        methods=self._get_all_methods()
        info={}
        _dfs=[]
        for m in methods:
            df=getattr(self.cls(),m)()
            _dfs.append(df)
            info[m.split('_')[1]]=df.columns.tolist()

        comb=join_dfs(_dfs)
        return comb,info

class Dataset:
    def __init__(self):
        self.data,self.info=self.get_data_and_info()

    def sample_control(self,df):
        '''
        this function is used to handle the sample problem,you can filter out financial stocks
        or you can set the time limit.
        :param df:
        :return:
        '''
        return df[df.index.get_level_values('t').year>=1996]

    def _combine(self):
        factor=Base(Unify_indicators)
        base=Base(Unify_base)
        info={**factor.info,**base.info}
        d_factor=factor.data
        d_factor=d_factor.shift(1)
        '''
            all the indicators are shift forward one month except for eret,rf and other base data,
        so the index denotes time t+1,and all the indicators are from time t,the base data are from 
        time t+1.We adjust the indicators rather than the base data for these reasons:
        1. we will sort the indicators in time t to construct portfolios and analyse the eret in time
            t+1
        2. We need to make sure that the index for eret and benchmark is corresponding to the time when 
        it was calcualted. If we shift back the base data in this place (rather than shift forward the
        indicators),we would have to shift forward eret again when we regress the portfolio eret on 
        benckmark model in the function _alpha in template.py
        '''
        d_base=base.data
        data=pd.concat([d_factor,d_base],axis=1)
        return data,info

    def get_data_and_info(self):
        # TODO:clear the .pkl files before runing the program
        p_data = os.path.join(TMP_PATH,'data.pkl')
        p_info = os.path.join(TMP_PATH,'info.pkl')
        if os.path.isfile(p_data) and os.path.isfile(p_info):
            with open(p_data, 'rb') as f:
                data = pickle.load(f)
            with open(p_info, 'rb') as f:
                info = pickle.load(f)
        else:
            data, info = self._combine()
            data=self.sample_control(data)
            pickle.dump(data, open(p_data, 'wb'))
            pickle.dump(info, open(p_info, 'wb'))

        return data,info

    @property
    def all_indicators(self):
        return sum(self.info.values(),[])

    def by_factor(self,factorname):
        return self.data[self.info[factorname]].dropna(how='all')

    def by_indicators(self,indicators):
        '''
        no mather indicators is just a string represent one indicators
        or list (tuple),the function will return a DataFrame
        :param indicators:
        :return: DataFrame
        '''
        if isinstance(indicators,(list,tuple)):
            return self.data[indicators].dropna(how='all')
        else:
            return self.data[[indicators]].dropna(how='all')

DATA=Dataset()

def save_info():
    ss=[pd.Series(v,name=k) for k,v in DATA.info.items()]
    df=pd.concat(ss,axis=1)
    df.to_csv('info.csv')


