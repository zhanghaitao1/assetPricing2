#-*-coding: utf-8 -*-
#author:tyhj
#assetPricing_Siriwardane.py 2017/8/14 18:55

from __future__ import division
from __future__ import print_function

import csv
import sys
import datetime
import re
import numpy as np
import copy
import cPickle
import math
from numpy import dot, mat, asarray, mean, size, shape, hstack, ones, ceil, \
    zeros, arange
from numpy.linalg import inv, lstsq
from scipy.stats import chi2
from scipy.stats import f as F_dist

r''' The empirical asset pricing (EAP) module is desiged to calculate the standard time-series and cross-sectional asset pricing tests that are used commonly in the empirical asset pricing literature.
	 An excellent reference for these methods are John Cochrane's PhD lecture notes.

	 In general, this module is designed to operate on a set of portfolios (N) and a set of factors (K).  The module will estimate pricing errors for individual portfolios and jointly test the hypothesis that
	 the pricing errors are zero.  Additionally, the module will estimate the price of risk associated with each factor using period by period cross-sectional estimates.  Finally, the code provides estimates
	 for the beta of each portfolio on the factor inputs

	 The time-series approach to empirical asset pricing assumes that the factors are tradeable assets.  In this case, a time-series regression for each of the N portfolios delivers estimates for the
	 pricing errors (alpha_i).  In addition, we are left with estimates of the beta's for each portfolio on each factor.  The price of risk for each factor is then simply its time series average.
	 The joint test that the pricing errors are zero is conducted using the stacked covariance of the time-series regression residuals.

	 The cross-sectional approach can be done in two ways.  I will start by using the Fama-MacBeth procedure and eventually add the one step pooled regression estimation to this code.  Fama-Macbeth starts by
	 estimating the beta's on each of the factors, portfolio by portfolio.  This is then N different time-series regressions.  Then, at each point in time, we use the estimated betas to run a cross-sectional
	 regression of portfolio returns on betas.  The slope coefficients in this regression is an estimate for the price of risk associated with each factor in that period.  If a constant is included, we do not
	 presume that the risk-free rate is the inverse of the expected value of the discount factor.  The regression residuals in each period, for each portfolio, represent the pricing errors for that period.
	 These will serve as the basis for the joint test that the pricing errors are non-zero

	 Note: The olsnw function for this code was taken from Kevin Sheppard's introduction to Python on his website and I recommend this great website for anyone trying to learn Python.

	 All errors are my own and please email me esiriwar@stern.nyu.edu with any errors.

	 Generic Inputs:
	 ---------------
	 	P = N x T array of portfolio returns.  Each row corresponds to the time-series of portfolio returns.
	 	F = K x T array of factors.  If the desired pricing test is the time-series method, the user must ensure these factors are traded


 	Copyright: Emil Siriwardane (esiriwar@stern.nyu.edu)
 			   NYU Stern School of Business
 			   May 9, 2013
'''

def time_series(P, F):
    r''' The time_series function of the EAP module will take a set of portfolios (NxT) and factors (KxT) and estimate the 1) betas for each portfolio on the factors, 2) the pricing errors for each
         of the portfolios, 3) the price of risk for each factor.

        Note: All standard errors and statistical inference is conducted using Newey-West (1987) corrected errors.
              This correction uses the generalization of the K-factor GMM procedure outlined in Cochrane's
              Asset Pricing Book  (Cochrane 2001).

        Parameters:
        ------------

        P : array_like
            The NxT array of portfolios.

        F : array_like
            The KxT array of factors.




        Returns:
        ------------

        alpha : ndarray, shape(N,1)
                Estimates of the pricing errors

        Beta  : ndarray, shape(N,K)
                Estimates of the beta of each portfolio on each factor

        Chi_stat 	  : scalar
                Test that all of the alphas are jointly zero.

        Usage:
        -------------
        alpha, t_alpha, beta, t_beta, Lambda, t_Lambda, Chi_stat, p_value, R2_all, R2bar_all = EAP.time_series(P,F,correction)

    '''

    ### Read in inputs
    N, T_P = shape(P)
    N = int(N)
    T_P = int(T_P)

    K, T_F = shape(F)
    K = int(K)
    T_F = int(T_F)

    if T_P != T_F:
        raise Exception('Time-series of portfolios must be same length as time-series of factors')
    else:
        T = T_P

    Lambda = np.mean(F, 1)
    if K > 1:
        Sigma_f = np.cov(F)
        t_Lambda = Lambda / np.sqrt(np.diag(Sigma_f) / T)
    elif K == 1:
        Sigma_f = np.var(F)
        t_Lambda = Lambda / np.sqrt(Sigma_f / T)

    Lambda = np.array([Lambda]).T

    ### Stack the portfolios to run stacked regression
    stacked_Y = np.ndarray.flatten(P)
    F_aug = np.hstack(
        (np.ones((T, 1)), F.T))  # Transpose the factors, and add a column of ones for the regression constant
    stacked_X = np.kron(np.eye(N), F_aug)

    stacked_b, stacked_vcv, stacked_s2, stacked_R2, stacked_R2bar, stacked_e = olsnw(stacked_Y, stacked_X,
                                                                                     constant=False)

    ### Put coefficients and errors into their own vectors
    alpha = np.empty((N, 1))
    beta = np.empty((N, K))

    Epsilon = np.empty((N, T))
    for p in range(0, N):
        alpha[p, :] = stacked_b[p * (K + 1)]
        beta[p, :] = np.ndarray.flatten(stacked_b[p * (K + 1) + 1:(p + 1) * (K + 1)])

        Epsilon[p, :] = np.ndarray.flatten(stacked_e[p * T:(p + 1) * T])

    #### Calculate standard errors ####
    Ktilde = K + 1
    moments = np.empty((N * (K + 1), T))

    for t in range(0, T):
        temp1 = np.reshape(Epsilon[:, t], (N, 1))
        temp2 = np.reshape(np.kron(Epsilon[:, t], F[:, t]), (N * K, 1))
        moments[:, t] = np.reshape(np.vstack((temp1, temp2)), (N * (K + 1),))

    # Demean moments (as per Cochrane suggestion)
    mean_moments = np.reshape(np.mean(moments, 1), (N * Ktilde, 1))
    mean_moments_mat = np.kron(np.ones((1, N * Ktilde)), mean_moments)
    moments = moments - mean_moments

    # Construct D matrix
    mean_F = np.array([np.mean(F, 1)]).T
    FFp = F.dot(F.T) / T
    D1 = np.hstack((np.eye(N), np.kron(np.eye(N), mean_F.T)))

    D = copy.deepcopy(D1)

    for i in range(0, N):
        io = iota(N, i + 1)
        D2_a = np.kron(io, mean_F)
        D2_b = np.kron(io, FFp)

        D2 = np.hstack((D2_a, D2_b))

        D = np.vstack((D, D2))

    # Construct S matrix
    lags = int(ceil(1.2 * float(T) ** (1.0 / 3)))
    w = 1 - arange(0, lags + 1) / (lags + 1)

    Gamma = zeros((lags + 1, N * Ktilde, N * Ktilde))
    for lag in xrange(lags + 1):
        Gamma[lag] = moments[:, lag:].dot(moments[:, :T - lag].T)

    Gamma = Gamma / T
    S = Gamma[0].copy()
    for i in xrange(1, lags + 1):
        S = S + w[i] * (Gamma[i] + Gamma[i].T)

    VCV = inv(D).dot(S).dot(inv(D).T) / T
    alpha_vcv = VCV[0:N, 0:N]
    beta_vcv = VCV[N:, N:]

    alpha_se = np.sqrt(np.diag(alpha_vcv))
    beta_se = np.sqrt(np.diag(beta_vcv))

    t_alpha = np.empty((N, 1))
    t_beta = np.empty((N, K))

    for p in range(0, N):
        t_alpha[p, :] = alpha[p, :] / alpha_se[p]
        t_beta[p, :] = beta[p, :] / beta_se[p * K:(p + 1) * K]

    # Joint test that all alpha are zero
    Chi_stat = float((alpha.T).dot(inv(alpha_vcv)).dot(alpha))
    p_value = 1 - chi2.cdf(Chi_stat, N)

    ### Calculate R2 for each time-series regression

    R2_all = np.empty((N, 1))
    R2bar_all = np.empty((N, 1))

    for p in range(0, N):
        Y = P[p, :]
        X = np.transpose(copy.copy(F))

        b, vcv, s2, R2, R2bar, e = olsnw(Y, X)
        R2_all[p, :] = R2
        R2bar_all[p, :] = R2bar

    # Output dictionary
    outputdict = {}
    outputdict['alpha'] = alpha
    outputdict['t_alpha'] = t_alpha
    outputdict['lamda'] = Lambda
    outputdict['t_lambda'] = t_Lambda
    outputdict['Chi_stat'] = Chi_stat
    outputdict['p_value'] = p_value
    outputdict['R2_all'] = R2_all
    outputdict['R2bar_all'] = R2bar_all
    outputdict['beta'] = beta
    outputdict['t_beta'] = t_beta

    return outputdict


# return alpha, t_alpha, beta, t_beta, Lambda, t_Lambda, Chi_stat, p_value, R2_all, R2bar_all

def GMM_cross_sectional(P, F, cs_constant=False, lags=None):
    r''' The GMM cross-sectional regression module will calculate the price of risk as well as conduct a test that the cross-sectional alphas are zero.
         The standard errors are corrected for generated regressors, heteroskedasticity, and autocorrelation.  The cross-sectional regressions are run as OLS,
         so that the parameter estimates match those that would be delivered by the Fama-Macbeth (1973) procedure.

        Parameters:
        ------------

        P : array_like
            The NxT array of portfolios.

        F : array_like
            The KxT array of factors.

        Returns:
        ------------

        alpha : ndarray, shape(N,1)
                Estimates of the pricing errors

        Beta  : ndarray, shape(N,K)
                Estimates of the beta of each portfolio on each factor

        lambda: ndarray, shape(K,1)
                Estimate for the price of risk of each factor

        F 	  : scalar
                F-test that all of the alphas are jointly zero.  Corrected for small samples

        Usage:
        -------------
        alpha, t_alpha, lam, t_lam, Jtest_stat, p_value, R2, R2adj, mape, beta = EAP.GMM_cross_sectional(P,F)



'''

    factors = F.T
    portfolios = P.T

    T, K = factors.shape
    T, N = portfolios.shape

    excessReturns = portfolios  # This is just for variable naming, I have already deducted the risk-free rate from the portfolios

    # Time series regressions
    beta = np.empty((N, K))

    Epsilon = np.empty((N, T))

    for p in range(0, N):
        Y = P[p, :]
        X = np.transpose(copy.copy(F))

        b, vcv, s2, R2, R2bar, e = olsnw(Y, X)
        tstats = b / np.sqrt(np.diag(vcv))

        beta[p, :] = np.ndarray.flatten(b[1:])
        Epsilon[p, :] = np.ndarray.flatten(e)

    # Add a constant (if specified for the cross-sectional regression)
    if cs_constant:
        beta_cs = np.hstack((np.ones((N, 1)), beta))
        K_cs = K + 1
    else:
        beta_cs = beta
        K_cs = K

    # Cross-section regression
    avgExcessReturns = np.mean(excessReturns, 0)

    # lam,vcv,s2,R2,R2bar,e = olsnw(avgExcessReturns,beta,constant=True)
    lam, vcv, s2, R2_g, R2bar, e = olsnw(avgExcessReturns, beta_cs, constant=False)

    R2 = 1 - np.var(e) / np.var(avgExcessReturns)
    R2adj = 1 - (np.var(e) / np.var(avgExcessReturns)) * ((N - 1) / (N - K - 1))

    lam = np.reshape(lam, (len(lam), 1))

    if cs_constant:
        lam_nc = lam[1:]
    else:
        lam_nc = lam

    # mean-absolute pricing error
    mape = np.abs(cs_constant * lam[0]) + np.mean(np.abs(e))

    ## Confidence intervals for adjusted R2


    # Get pricing errors
    alphas = np.empty((N, T))

    for t in range(0, T):
        alphas[:, t] = P[:, t] - np.ndarray.flatten(beta_cs.dot(lam))

    #### Calculate standard errors
    Ktilde = K + 1
    moments = np.empty((N * (K + 1) + N, T))

    for t in range(0, T):
        temp1 = np.reshape(Epsilon[:, t], (N, 1))
        temp2 = np.reshape(np.kron(Epsilon[:, t], F[:, t]), (N * K, 1))
        temp3 = alphas[:, t]
        m1 = np.reshape(np.vstack((temp1, temp2)), (N * (K + 1),))
        moments[:, t] = np.hstack((m1, alphas[:, t]))

    # Demean moments (as per Cochrane suggestion)
    mean_moments = np.reshape(np.mean(moments, 1), (N * Ktilde + N, 1))
    mean_moments_mat = np.kron(np.ones((1, N * Ktilde + N)), mean_moments)
    moments = moments - mean_moments

    # Construct 'a' matrix
    a1 = np.hstack((np.eye(N * Ktilde), np.zeros((N * Ktilde, N))))
    a2 = np.hstack((np.zeros((K_cs, N * Ktilde)), beta_cs.T))
    a = np.vstack((a1, a2))

    # Construct D matrix
    mean_F = np.array([np.mean(F, 1)]).T
    FFp = F.dot(F.T) / T
    D1 = np.hstack((np.eye(N), np.kron(np.eye(N), mean_F.T)))

    D_ul = copy.deepcopy(D1)

    for i in range(0, N):
        io = iota(N, i + 1)
        D2_a = np.kron(io, mean_F)
        D2_b = np.kron(io, FFp)

        D2 = np.hstack((D2_a, D2_b))

        D_ul = np.vstack((D_ul, D2))

    D_ur = np.zeros((N * (K + 1), K_cs))

    D_top = np.hstack((D_ul, D_ur))

    D_bottom = np.hstack((np.zeros((N, N)), np.kron(np.eye(N), lam_nc.T)))
    D_bottom = np.hstack((D_bottom, beta_cs))

    D = -np.vstack((D_top, D_bottom))

    # Construct S matrix
    if lags == None:
        lags = int(ceil(1.2 * float(T) ** (1.0 / 3)))
    w = 1 - arange(0, lags + 1) / (lags + 1)

    Gamma = zeros((lags + 1, N * Ktilde + N, N * Ktilde + N))
    for lag in xrange(lags + 1):
        Gamma[lag] = moments[:, lag:].dot(moments[:, :T - lag].T)

    Gamma = Gamma / T
    S = Gamma[0].copy()
    for i in xrange(1, lags + 1):
        S = S + w[i] * (Gamma[i] + Gamma[i].T)

    adi = inv(a.dot(D))
    VCV = adi.dot(a).dot(S).dot(a.T).dot(adi.T) / T

    # Covariance matrix of scores
    iminusad = np.eye(N * Ktilde + N) - D.dot(adi).dot(a)
    VCV_scores = iminusad.dot(S).dot(iminusad.T) / T

    ### Compute standard errors of lambdas
    se_lam = np.sqrt(np.diag(VCV[-K_cs:, -K_cs:]))
    t_lam = (lam.T / se_lam).T

    ### Compute alphas and standard errors of alphas
    alpha = np.reshape(np.mean(alphas, 1), (N, 1))

    VCV_alpha = VCV_scores[-N:, -N:]
    se_alpha = np.sqrt(np.diag(VCV_alpha))
    t_alpha = (alpha.T / se_alpha).T

    Jtest_stat = (alpha.T).dot(inv(VCV_alpha)).dot(alpha)
    p_value = 1 - chi2.cdf(Jtest_stat, N - K_cs)

    return alpha, t_alpha, lam, t_lam, Jtest_stat, p_value, R2, R2adj, mape, beta


def Fama_MacBeth(P, F, cs_constant=False):
    r''' The Fama-MacBeth module will run the standard Fama-Macbeth procedure on the portfolios and factors that are input.  The Fama-Macbeth procedure is run with a constant.


        Parameters:
        ------------

        P : array_like
            The NxT array of portfolios.

        F : array_like
            The KxT array of factors.

        Returns:
        ------------

        alpha : ndarray, shape(N,1)
                Estimates of the pricing errors

        Beta  : ndarray, shape(N,K)
                Estimates of the beta of each portfolio on each factor

        lambda: ndarray, shape(K,1)
                Estimate for the price of risk of each factor

        F 	  : scalar
                F-test that all of the alphas are jointly zero.  Corrected for small samples

        Usage:
        -------------
        alpha, t_alpha, beta, t_beta, Lambda, t_Lambda, Chi_test, p_value, R2, R2adj = EAP.Fama_MacBeth(P,F)

'''

    ### Read in inputs
    N, T_P = shape(P)
    N = int(N)
    T_P = int(T_P)

    K, T_F = shape(F)
    K = int(K)
    T_F = int(T_F)

    if T_P != T_F:
        raise Exception('Time-series of portfolios must be same length as time-series of factors')
    else:
        T = T_P

    # average excess returns
    exRet = np.mean(P, 1)

    ### Run a series of N time-series regressions

    beta = np.empty((N, K))
    t_beta = np.empty((N, K))
    epsilon = np.empty((N, T))

    for p in range(0, N):
        Y = P[p, :]
        X = np.transpose(copy.copy(F))

        b, vcv, s2, R2, R2bar, e = olsnw(Y, X)
        tstats = b / np.sqrt(np.diag(vcv))

        beta[p, :] = np.ndarray.flatten(b[1:])
        t_beta[p, :] = tstats[1:]
        epsilon[p, :] = np.ndarray.flatten(e)

    ### Run T different cross-sectional regressions
    if cs_constant:
        K_cs = K + 1
    else:
        K_cs = K

    price_risk = np.empty((T, K_cs))
    pricing_errors = np.empty((T, N))  # Each row is a pricing error, will transpose later

    for t in range(0, T):
        R = P[:, t]

        lambda_temp, vcv2, s2_2, R2_2, R2bar_2, e_temp = olsnw(R, beta, constant=cs_constant)
        # price_risk[t,:]=lambda_temp[1:]
        price_risk[t, :] = np.ndarray.flatten(lambda_temp)
        pricing_errors[t, :] = np.ndarray.flatten(e_temp)

    lam = np.reshape(np.mean(price_risk, 0).T, (K_cs, 1))

    if not cs_constant and K == 1:
        vcv_lam = np.cov(price_risk.T) / T
    else:
        vcv_lam = np.diag(np.cov(price_risk.T)) / T

    t_lam = np.reshape(lam.T / np.sqrt(vcv_lam), (K_cs, 1))

    # Shanken standard errors for the fact that betas are estimated
    if cs_constant:
        beta_aug = np.hstack((np.ones((N, 1)), beta))
    else:
        beta_aug = beta
    A = inv(beta_aug.T.dot(beta_aug)).dot(beta_aug.T)
    sh_sigma = A.dot(np.cov(epsilon)).dot(A.T)

    if cs_constant:
        if K == 1:
            c = float(1 + lam[1, 0] * ((np.cov(F)) ** (-1)) * (lam[1, 0]))
        else:
            c = float(1 + lam[1:].T.dot(inv(np.cov(F))).dot(lam[1:]))
    elif not cs_constant and K == 1:
        c = float(1 + lam[0:].T.dot((np.cov(F)) ** (-1)).dot(lam[0:]))
    else:
        c = float(1 + lam[0:].T.dot(inv(np.cov(F))).dot(lam[0:]))

    if cs_constant:
        F_aug = np.vstack((np.ones((1, T)), F))
    else:
        F_aug = F
    vcv_shank_lam = np.diag((c * sh_sigma + np.cov(F_aug)) / T)

    t_lam_shank = np.reshape(lam.T / np.sqrt(vcv_shank_lam), (K_cs, 1))

    alpha = np.array([np.mean(pricing_errors, 0)]).T
    Sigma_alpha = np.cov(pricing_errors.T) / T

    t_alpha = (alpha.T / np.sqrt(np.diag(Sigma_alpha))).T

    Chi_test = float((alpha.T).dot(inv(Sigma_alpha)).dot(alpha))
    p_value = 1 - chi2.cdf(Chi_test, N - K)

    # average excess returns for R2
    exRet = np.mean(P, 1)
    lambda_temp, vcv2, s2_2, R2_2, R2bar_2, e_temp = olsnw(exRet, beta, constant=cs_constant)

    R2 = 1 - np.var(e_temp) / np.var((exRet))
    R2adj = 1 - np.var(alpha) / np.var(exRet) * ((N - 1) / (N - K - 1))

    mape = np.abs(cs_constant * lam[0]) + np.mean(np.abs(alpha))

    return alpha, t_alpha, beta, t_beta, lam, t_lam, t_lam_shank, Chi_test, p_value, R2, R2adj, mape


def olsnw(y, X, constant=True, lags=None):
    T = y.size
    if size(X, 0) != T:
        X = X.T
        T, K = shape(X)
    if constant:
        X = copy.copy(X)
        X = hstack((ones((T, 1)), X))
        K = size(X, 1)

    K = size(X, 1)
    if lags == None:
        lags = int(ceil(1.2 * float(T) ** (1.0 / 3)))
    # Parameter estimates and errors
    out = lstsq(X, y)
    b = out[0]
    e = np.reshape(y - dot(X, b), (T, 1))
    # Covariance of errors

    gamma = zeros((lags + 1))
    for lag in xrange(lags + 1):
        gamma[lag] = e[:T - lag].T.dot(e[lag:]) / T
    w = 1 - arange(0, lags + 1) / (lags + 1)
    w[0] = 0.5

    s2 = dot(gamma, 2 * w)
    # Covariance of parameters
    Xe = mat(zeros(shape(X)))
    for i in xrange(T):
        Xe[i] = X[i] * float(e[i])
    Gamma = zeros((lags + 1, K, K))
    for lag in xrange(lags + 1):
        Gamma[lag] = Xe[lag:].T * Xe[:T - lag]

    Gamma = Gamma / T
    S = Gamma[0].copy()
    for i in xrange(1, lags + 1):
        S = S + w[i] * (Gamma[i] + Gamma[i].T)
    XpX = dot(X.T, X) / T
    XpXi = inv(XpX)
    vcv = mat(XpXi) * S * mat(XpXi) / T
    vcv = asarray(vcv)
    # R2, centered or uncentered
    if constant:
        R2 = e.T.dot(e) / ((y - mean(y)).T.dot(y - mean(y)))
    else:
        R2 = e.T.dot(e) / ((y.T).dot(y))

    R2bar = 1 - R2 * (T - 1) / (T - K)
    R2 = 1 - R2
    return b, vcv, s2, R2, R2bar, e


def iota(N, i):
    if i > N:
        raise Exception('Index can not be longer than vector')
    temp = np.zeros((1, N))
    temp[0, i - 1] = 1
    return temp





















