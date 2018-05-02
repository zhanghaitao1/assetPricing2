# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-02  08:54
# NAME:assetPricing2-analyse.py
import os

from core.main import Bivariate




class Test(Bivariate):
    def __init__(self):
        indicator1='a' #no use
        indicator2='b' # no use
        path=os.path.join(r'D:\zht\database\quantDb\researchTopics\assetPricing2\lewellen')
        super().__init__(indicator1,indicator2,path)

    #TODO: how to send paramters into a class method without inherit and overide the method
    def fm(self):
        l = ['beta__M_36M', 'size__size', 'value__logbm', 'momentum__R3M', 'reversal__reversal', 'liquidity__amihud',
             'skewness__coskew_6M__D',
             'idio__volss_6M__D']
        super()._fm(l)

    def run(self):
        self.fm()

    def __call__(self):
        self.run()





if __name__ == '__main__':
    Test()()


