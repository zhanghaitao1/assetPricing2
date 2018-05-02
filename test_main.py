# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-04-01  11:31
# NAME:assetPricing2-test_main.py

from old.main import OneFactor, Bivariate

def test_Onefactor():
    # factor = 'beta'
    # path = r'D:\zht\database\quantDb\researchTopics\assetPricing2\beta'
    # beta = OneFactor(factor, path)
    # beta()

    factor='size'
    path=r'D:\zht\database\quantDb\researchTopics\assetPricing2\size'
    size=OneFactor(factor,path)
    size()

    # factor='value'
    # path=r'D:\zht\database\quantDb\researchTopics\assetPricing2\value'
    # value=OneFactor(factor,path)
    # value()
    #
    # factor = 'momentum'
    # path = r'D:\zht\database\quantDb\researchTopics\assetPricing2\mom'
    # mom = OneFactor(factor, path)
    # mom()
    #
    # factor='reversal'
    # path=r'D:\zht\database\quantDb\researchTopics\assetPricing2\reversal'
    # rev=OneFactor(factor,path)
    # rev()
    #
    # factor='skewness'
    # path=r'D:\zht\database\quantDb\researchTopics\assetPricing2\skewness'
    # skewness=OneFactor(factor,path)
    # skewness()
    #
    # factor='idiosyncraticVolatility'
    # path=r'D:\zht\database\quantDb\researchTopics\assetPricing2\idiosyncraticVolatility'
    # idiosyn=OneFactor(factor,path)
    # idiosyn()


class size_12M(Bivariate):
    def __init__(self):
        indicator1='size'
        indicator2='D_12M'
        path=r'D:\zht\database\quantDb\researchTopics\assetPricing2\size_12M'
        super().__init__(indicator1,indicator2,path)

    #TODO: how to send paramters into a class method without inherit and overide the method
    def fm(self):
        ll_indeVars = [['bm'], ['bm', 'M_12M'], ['bm', 'size'], ['bm', 'M_12M', 'size'],
                       ['logbm'], ['logbm', 'M_12M'], ['logbm', 'size'], ['logbm', 'M_12M', 'size']]
        super()._fm(ll_indeVars)

    def run(self):
        # self.dependent_portfolio_analysis()
        # self.independent_portfolio_analysis()
        self.fm()

    def __call__(self):
        self.run()

def test_Bivariate():
    size_12M()()

if __name__=='__main__':
    test_Onefactor()
    # test_Bivariate()
