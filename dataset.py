# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-03-22  15:15
# NAME:assetPricing2-dataset.py
from types import FunctionType

from config import DATA_PATH, TMP_PATH
from dout import read_df
import pandas as pd
import os
import pickle
from pandas.tseries.offsets import MonthEnd
import datetime

from zht.utils.dfu import join_dfs

def unify_df(df):
    '''
    This function has the following features:
    1. detect the invalid index format and make them consistent with my standard

    :param df:
    :return:
    '''
    names=df.index.names

    df=df.reset_index()
    for name in names:
        if name=='t':
            v=df['t'][0]
            if not isinstance(v,pd.Timestamp):
                raise TypeError('The format of index for level t is not pd.Timesamp !! ')
        if name=='sid':
            v=df['sid'][0]
            if not isinstance(v,str):
                df['sid']=df['sid'].astype(str)
                print('Converted the format of index for level "sid" from {} to "str"'.format(type(v)))
    df=df.set_index(names,drop=True)
    return df


class T_indicators:
    '''
    the indicators in this class is the data we get in time t

    '''

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
        capM=read_df('capM','M')
        size=read_df('size','M')
        mktCap_ff=read_df('mktCap_ff','M')
        size_ff=read_df('size_ff','M')
        #index:t
        #columns:sid
        capM=capM.stack()
        capM.name='mktCap'
        size=size.stack()
        size.name='size'
        mktCap_ff=mktCap_ff.stack()
        mktCap_ff.name='mktCap_ff'
        size_ff=size_ff.stack()
        size_ff.name='size_ff'
        comb=pd.concat([capM,size,mktCap_ff,size_ff],axis=1)
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
        illiq=illiq.stack().unstack('type')
        illiq.index.names=['t','sid']

        liqBeta=read_df('liqBeta','M')
        liqBeta=liqBeta.stack()
        liqBeta.index.names=['t','sid']
        liqBeta.name='liqBeta'

        comb=pd.concat([illiq,liqBeta],axis=1)
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
        comb=pd.concat(dfs,axis=1)
        return comb

    def unify_capM(self):
        '''
        market capitalization

        Usually,the market capitalization is used as weight and we use this value at time t

        :return:
        '''
        capM=read_df('capM','M')
        capM=capM.stack().to_frame()
        capM.index.name='t'
        capM.columns=['capM']
        capM.index.names=['t','sid']
        return capM

def _add_prefix(df,prefix):
    oldCol=df.columns
    newCol=['__'.join([prefix,col]) for col in oldCol]
    df.columns=newCol
    return df


class Benchmark:
    def __init__(self):
        self.data=self._get_data()
        self.models=['capm','ff3','ffc','ff5','hxz4']

    #--------------------------
    #benchmark model with single index
    def unify_capmM(self):
        capmM=read_df('rpM','M')
        capmM=_add_prefix(capmM,'capmM')
        return capmM

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

    def _get_data(self):
        capm=self.unify_capmM()
        ff3=self.unify_ff3M()
        ffc=self.unify_ffcM()
        ff5=self.unify_ff5M()
        hxz4=self.unify_hxz4M()

        data={'capm':capm,
              'ff3':ff3,
              'ffc':ffc,
              'ff5':ff5,
              'hxz4':hxz4}

        return data

BENCH=Benchmark()

class T1_indicators:
    '''
    This class contains the indicators related to prediction process.We can not get
    these indicators in time t and we will use those indicators we can get in time
    t (refer to class T_indicators) to predict these indicators,such as eretM.

    '''

    #------------------------
    #multiIndex
    def unify_eretM(self):
        #eretM
        eretM=read_df('eretM','M')
        eretM=eretM.stack().to_frame()
        eretM.columns=['eretM']
        eretM.index.names=['t','sid']
        return eretM

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
            df=unify_df(df) # some of the DataFrame do not share the same index format
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
        :param df:multiIndex DataFrame
        :return:
        '''
        '''
        use truncating  & fancy indexing
        refer to http://pandas.pydata.org/pandas-docs/stable/timeseries.html#
        
        '''
        return df[df.index.get_level_values('t').year>=1996]

    def _combine(self):
        factor=Base(T_indicators)
        base=Base(T1_indicators)
        info={**factor.info,**base.info}
        d_factor=factor.data
        # For multiIndex DataFrame,when we want to shift on a given index,we should
        # use groupby(indexname).shift()
        d_factor=d_factor.groupby('sid').shift(1)

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
        
        For simply,we use the information at t to predict the eret of time t+1.In our DATA.data,the index
        denotes time t,and the values for eretM,benchmark model and so on is from time t+1.
        
        Notice:
            To calculate value-weighted result,we use the market capitalization of the time t (portfolio
            formation period) as weight.So,in this place,we should shift the capM forward for one step
            as well.For more details,refer to page 40 of Bali.
        
        '''
        d_base=base.data
        data=pd.concat([d_factor,d_base],axis=1)
        return data,info

    def get_data_and_info(self):
        # TODO:clear the .pkl files before runing the program
        p_data = os.path.join(TMP_PATH,'data.pkl')
        p_info = os.path.join(TMP_PATH,'info.csv.pkl')
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
            #TODO: use df.to_pickle and pd.read_pickle

        return data,info

    def by_factor(self,factorname):
        return self.data[self.info[factorname]].copy(deep=True).dropna(how='all')

    def by_indicators(self,indicators):
        '''
        no mather indicators is just a string represent one indicators
        or list (tuple),the function will return a DataFrame
        :param indicators:
        :return: DataFrame
        '''
        if isinstance(indicators,(list,tuple)):
            return self.data[list(indicators)].copy(deep=True).dropna(how='all')
        else:
            return self.data[[indicators]].copy(deep=True).dropna(how='all')

def save_info():
    DATA=Dataset()
    ss=[pd.Series(v,name=k) for k,v in DATA.info.items()]
    df=pd.concat(ss,axis=1)
    df.to_csv('info.csv.csv')


#--------------------------------------------sample control----------------------------------------------
#TODO: sample control should be put before calculating the indicators or factors
def control_stock_sample(df, condition):
    '''
    is_sz
    is_sh
    is_gem 创业板
    is_cross
    not_financial
    is_industry

    :param df:
    :param condition:
    :return:stock code
    '''
    #TODO: not_financial 保险？
    conditions=['is_sz','is_sh','not_cross','not_financial']

    info=pd.read_csv(os.path.join(DATA_PATH,'listInfo.csv'),index_col=0,parse_dates=True)
    info.index=info.index.astype(str)

    if condition in conditions:
        validSids=info.index[info[condition]]
    else:
        raise ValueError('The "condition" should be one of {}'.format(repr(conditions)))

    newdf=df.loc[pd.IndexSlice[:,validSids],:]

    return newdf


def start_end(df, start='1996-01-01', end=None):
    '''



    start
    end

    is_bear
    is_bull
    is_downside?

    :return:
    '''
    if isinstance(start,str):
        start=pd.to_datetime(start)

    if not end:
        end=datetime.datetime.today()

    if isinstance(end,str):
        end=pd.to_datetime(end)

    if 't' in df.index.names:
        return df[(start<=df.index.get_level_values('t')) &(df.index.get_level_values('t')<=end)]
    elif 't' in df.columns:
        return df[(start<=df['t'])&(df['t']<=end)]
    else:
        raise ValueError('There is no index or column named "t" !')



def sample_data_optimization():
    '''

    '''
    pass


def filter_multiIndex(df,inter_index):
    df=df.reindex(inter_index)
    df=df.dropna(how='all')
    return df

def floor_price(df,clsPrice=5.0):
    '''
    the minimum close price is 5

    :param df:
    :param clsPrice:
    :return:
    '''
    stockCloseM=read_df('stockCloseM','M')
    stockCloseM.columns=stockCloseM.columns.astype(str)

    valid=stockCloseM[stockCloseM>=5.0].stack()
    df=filter_multiIndex(df,valid.index)
    return df


def roof_price(df,price):
    pass

def in_event_window(df):
    pass


#TODO: refer to readme.md to find more controling methods.
def year_after_list(df,freq):
    '''
    listed at list 1 year

    :return:
    '''
    listInfo=pd.read_csv(os.path.join(DATA_PATH,'listInfo.csv'),index_col=0)
    listInfo.index=listInfo.index.astype(str) #TODO: delete
    listInfo['listDate']=pd.to_datetime(listInfo['listDate'])
    listInfo['year_later']=listInfo['listDate']+pd.offsets.DateOffset(years=1)
    if freq=='M':
        listInfo['year_later']=listInfo['year_later']+MonthEnd(1)
        # 1 rather than 0,exclude the first month,since most of
        # year_later won't be monthend.
    elif freq=='D':
        listInfo['year_later']=listInfo['year_later']+pd.offsets.DateOffset(days=1)

    sids=df.index.get_level_values('sid').unique()
    start=df.index.get_level_values('t').min()

    yearlater=listInfo['year_later']

    sid_inter=list(set(sids).intersection(set(yearlater.index.tolist())))
    valid=pd.DataFrame(columns=sid_inter,index=pd.date_range(
        start=start,end=datetime.datetime.today().strftime('%Y-%m-%d'),freq=freq))

    yearlater=listInfo['year_later']
    for sid,d in yearlater.iteritems():
        if d<start:
            d=start
        valid.at[d,sid]=True

    valid=valid.fillna(method='ffill')
    valid=valid.stack()
    valid.index.names=['t','sid']

    df=filter_multiIndex(df,valid.index)
    return df

#TODO: din.py get_sid_suffix()

def delete_st(df,freq='M'):
    if freq=='M':
        stInfo=pd.read_csv(os.path.join(DATA_PATH,'stInfoM.csv'),index_col=[0,1],parse_dates=True)
    elif freq=='M':
        stInfo=pd.read_csv(os.path.join(DATA_PATH,'stInfoD.csv'),index_col=[0,1],parse_dates=True)
    else:
        raise ValueError('freq must be "M" or "D"!')

    stInfo=stInfo.reset_index()

    stInfo['sid']=stInfo['sid'].astype(str)
    stInfo=stInfo.set_index(['t','sid'])

    # filter
    df=filter_multiIndex(df,stInfo.index)
    return df

def _get_DATA():
    DATA=Dataset()
    #TODO: use pipe
    print(DATA.data.shape)
    DATA.data=control_stock_sample(DATA.data, 'not_financial')
    print(DATA.data.shape)
    DATA.data=control_stock_sample(DATA.data,'not_cross')
    print(DATA.data.shape)
    DATA.data=start_end(DATA.data, start='2001-01-01')
    print(DATA.data.shape)
    DATA.data=year_after_list(DATA.data,freq='M')
    print(DATA.data.shape)
    DATA.data=delete_st(DATA.data,freq='M')
    print(DATA.data.shape)
    DATA.data=floor_price(DATA.data,5.0)
    print(DATA.data.shape)
    return DATA

def get_DATA():
    p_tmp_DATA=os.path.join(TMP_PATH,'tmp_DATA.pkl')
    if os.path.isfile(p_tmp_DATA):
        with open(p_tmp_DATA,'rb') as f:
            DATA=pickle.load(f)
    else:
        DATA=_get_DATA()
        pickle.dump(DATA,open(p_tmp_DATA,'wb'))
    return DATA

DATA=get_DATA()





#TODO: sid add suffix


#TODO:The 10% limit policy took effect at the beginning of 1997.We exclude stocks that have been listed for less than one year and returns on the first day after the initial public offering.
#TODO: as robustness check in Long, Jiang, and Zhu, “Idiosyncratic Tail Risk and Expected Stock Returns: Evidence from the Chinese Stock Markets.”
'''
Delete ST (special treatment) and firms in financial industry.
    How to delete ST,delete from the ST date or delete the ST stocks from the whole sample?

susamples,before and after December 2006.
subsamples with returns higher and lower than the median index return
subsamples before and after March 2010 when Chinese stock markets partially allowed short sales.
calculate portfolio returns with different holding epriods of 2,6,and 12 months
'''

