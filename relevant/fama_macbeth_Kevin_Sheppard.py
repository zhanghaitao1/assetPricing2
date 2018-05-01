#-*-coding: utf-8 -*-
#author:tyhj
#fama_macbeth_Kevin_Sheppard.py 2017.10.23 12:03

from numpy import mat, cov, mean, hstack, multiply,sqrt,diag, \
squeeze, ones, array, vstack, kron, zeros, eye, savez_compressed
from numpy.linalg import inv
from scipy.stats import chi2
from pandas import read_csv
import statsmodels.api as sm



data = read_csv(r'C:\Python27\zht\util\assetPricing\FamaFrench.csv')



#split using both named columns and ix for larger blocks
dates=data['date'].values
factors=data[['VWMe','SMB','HML']].values
riskfree=data['RF'].values
portfolios=data.ix[:,5:].values

#use mat for easier linear algebra
factors=mat(factors)
riskfree=mat(riskfree)
portfolios=mat(portfolios)

#shape information
T,K=factors.shape
T,N=portfolios.shape
#reshape rf and compute excess returns
riskfree.shape=T,1
excessReturns=portfolios-riskfree

#time series regressions
X=sm.add_constant(factors)
ts_res=sm.OLS(excessReturns,X).fit()
alpha=ts_res.params[0]
beta=ts_res.params[1:]
avgExcessReturns=mean(excessReturns,0)
#cross-section regression
cs_res=sm.OLS(avgExcessReturns.T,beta.T).fit()
riskPremia=cs_res.params

# Moment conditions
X = sm.add_constant(factors)
p = vstack((alpha, beta))
epsilon = excessReturns - X @ p
moments1 = kron(epsilon, ones((1, K + 1)))
moments1 = multiply(moments1, kron(ones((1, N)), X))
u = excessReturns - riskPremia[None,:] @ beta
moments2 = u * beta.T








