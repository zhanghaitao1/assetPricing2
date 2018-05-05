# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-28  15:46
# NAME:assetPricing2-test_main.py
import os

from core.main import OneFactor,Bivariate

# testdir = r'D:\zht\database\quantDb\researchTopics\assetPricing2\20180427'
testdir=r'D:\zht\database\quantDb\researchTopics\assetPricing2\apply_condition_result'



def test_OneFactor():
    factors=['beta','size','value','momentum','reversal','liquidity','skewness','idio']

    for factor in factors: #TODO:
        path=os.path.join(testdir,factor)
        OneFactor(factor,path)()
        print(factor,'Finished')

class size_12M(Bivariate):
    def __init__(self):
        indicator1='size'
        indicator2='D_12M'
        path=os.path.join(testdir,'test_lewellen')
        super().__init__(indicator1,indicator2,path)

    #TODO: how to send paramters into a class method without inherit and overide the method
    def famaMacbeth(self):

        ll_indeVars = [['value__bm'], ['value__bm', 'beta__M_12M'], ['value__bm', 'size__size'],
                       ['value__bm', 'beta__M_12M', 'size__size'],
                       ['value__logbm'], ['value__logbm', 'beta__M_12M'],
                       ['value__logbm', 'size__size'], ['value__logbm', 'beta__M_12M', 'size__size']]
        super()._fm(ll_indeVars)

    def run(self):
        self.dependent_portfolio_analysis()
        self.independent_portfolio_analysis()
        self.fm()

    def __call__(self):
        self.run()

def test_Bivariate():
    size_12M()()


if __name__ == '__main__':
    test_OneFactor()
    # test_Bivariate()


