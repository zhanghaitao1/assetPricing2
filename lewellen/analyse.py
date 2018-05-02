# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-02  08:54
# NAME:assetPricing2-analyse.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from core.main import Bivariate
from data.dataTools import load_data
from tool import summary_statistics, newey_west
from zht.utils.mathu import get_inter_frame
import statsmodels.formula.api as sm

directory=r'D:\zht\database\quantDb\researchTopics\assetPricing2\lewellen'
l = ['beta__M_36M', 'size__size', 'value__logbm', 'momentum__R3M',
     'reversal__reversal', 'liquidity__amihud',
     'skewness__coskew_6M__D',
     'idio__volss_6M__D']
fn=os.path.join(directory,r'first stage parameters beta__M_36M_size__'
                              r'size_value__logbm_momentum__R3M_reversal__'
                              r'reversal_liquidity__amihud_skewness__coskew_'
                              r'6M__D_idio__volss_6M__D.csv')

class Test(Bivariate):
    def __init__(self):
        indicator1='a' #no use
        indicator2='b' # no use
        path=os.path.join(directory)
        super().__init__(indicator1,indicator2,path)

    #TODO: how to send paramters into a class method without inherit and overide the method
    def fm(self):
        super()._fm(l)

    def run(self):
        self.fm()

    def __call__(self):
        self.run()

def rolling_slopes():
    params=pd.read_csv(fn,index_col=0)
    params.index=pd.to_datetime(params.index)
    for col,s in params.iteritems():
        fig=plt.figure()
        s.rolling(12).mean().plot().get_figure()
        fig.savefig(os.path.join(directory,col+'.png'))



'''
The params means that these parameters are estimated by using the characteristics
of time t and return of time t+1.The index for params denotes time t+1.

On the other hand,the data indicators get from load_data has been shifted 
forward.So,we have to multiply the params of time t+1 with characteristics of
time t+2.That is,we have to shift backward the indicators loaded from function
"load_data" for 2 month.

'''
params=pd.read_csv(fn,index_col=0,parse_dates=True)
params=params.shift(1)
indicators=load_data('data')[l]
indicators['Intercept']=1.0
cols=params.columns
indicators=indicators.reindex(columns=cols)

#TODO: predict return rather than eret,
groups=list(indicators.groupby('sid'))
ss=[]
names=[]
for name,g in groups:
    g=g.reset_index(level='sid',drop=True)
    p,g=get_inter_frame([params,g.dropna()])
    s=(p * g).sum(axis=1)
    ss.append(s)
    names.append(name)
    print(name)

predicted=pd.concat(ss,axis=1,keys=names)
stockEret=load_data('stockEretM')
predicted,stockEret=get_inter_frame([predicted,stockEret])


months=[]
models=[]
for month,p in predicted.iterrows():
    r=stockEret.loc[month]
    df=pd.concat([p,r],axis=1,keys=['predicted','realized'])
    model=sm.ols(formula='realized ~ predicted',data=df).fit(use_t=True)
    months.append(month)
    models.append(model)
    print(month)

slope=pd.Series([m.params['predicted'] for m in models],index=months)
r2=pd.Series([m.rsquared for m in models],index=months)

newey_west(format=)

slope.head()
slope.describe()



#TODO: handle the time problem predict or



# fn=os.path.join(directory,'first stage fittedvalues beta__M_36M_size__size_value__logbm_momentum__R3M_reversal__reversal_liquidity__amihud_skewness__coskew_6M__D_idio__volss_6M__D.csv')
# df=pd.read_csv(fn,index_col=0)
# stat=summary_statistics(df).mean()
#
# stockEret=load_data('stockEretM')
#
# df=df.stack()
# stockEret=stockEret.stack()
# comb=pd.concat([df,stockEret],axis=1,keys=[''])
# comb.head()



# if __name__ == '__main__':
#     Test()()


