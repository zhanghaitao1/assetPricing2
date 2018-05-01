# -*-coding: utf-8 -*-
# Python 3.6
# Author:Zhang Haitao
# Email:13163385579@163.com
# TIME:2018-05-01  16:09
# NAME:assetPricing2-reg.py


import numpy as np
from numpy import ceil,argsort,zeros, eye,reshape, kron,dot,var,exp,mean,squeeze
from numpy.linalg import lstsq, inv, pinv, solve, eigh
from numpy.linalg import matrix_rank as rank
from pandas import read_csv
from scipy.stats import f, chi2,t
import statsmodels.api as sm
import ENSm as ENS

class regress(object):
    def __init__(self,y,X):
        """
        N = number of dependent variables
        T = number of data points
        k = number of factors
        yp = portfolios (not excess returns)
        y = dependent variable and has shape (T,N)
        X = independent variable and has shape (T,k)
        """
        # Set shape vars

        self.T = y.shape[0]
        self.N = y.shape[1]
        self.k = X.shape[1]
        self.nmom = self.N + self.N*self.k + self.N

        # Set endog and exog vars

        self.y = y
        self.X = X
        self.fac = X.T #(the transpose is so we can work with column vectors)

        # Calculate vcv of factors to correct for autocorrelation and heteroscedacity

        self.fm = reshape(mean(self.fac,axis=1),(self.k,1))
        self.vcvfac = dot(self.fac-self.fm,(self.fac - self.fm).T)/self.T

        # Calculate mean returns for CS regressions

        self.ym = y.mean(axis=0)

        #Setup identity matrices so we dont have to initialise them every time

        self.Ik = eye(self.k)
        self.IN = eye(self.N)
        self.INT = eye(self.N*self.T)

        # TS regression with an intercept

        self.thetaOLS, self.Sigmai, self.uti = self.TS_OLS(const=True)
        self.TSalpha = self.thetaOLS[0:self.N]
        self.TSbeta = reshape(self.thetaOLS[self.N::],(self.N,self.k)).T

        # Fama-Macbeth regression

        alpha,lamda,alphat,lambdat,vcvalpha,vcvlamb,R2,R2adj,ts_fmb,ts_pvalue = self.fmb(self.y,self.TSbeta)
        self.tsLamFMB = ts_fmb
        self.pvLamFMB = ts_pvalue
        self.fmb_pval = self.fmb_pval(alpha,vcvalpha)

        # TS regressions without an intercept

        self.betaOLS, self.Sigmani, self.utni = self.TS_OLS()
        self.Sigma = self.Sigmai
        #m = int(ceil(1.2 * float(self.T)**(1.0/3)))
        #self.Sigma = self.set_S(m,self.utni.T) #In case we want a weighting matrix
        if rank(self.Sigma,tol=1e-9) < self.N:
            print ("Warning Sigma has deficient rank of ", rank(self.Sigma,tol=1e-9))
        Bi = self.TSbeta.T
        Bni = reshape(self.betaOLS,(self.N,self.k))

        # Compute GRS test on TS regression

        self.GRStest = self.GRS_test(self.TSalpha,self.Sigma)

        # GMM regression with parameters estimated using the standard regression but errors
        # ected for using a HAC matrix

        self.do_TS_GMM(self.uti.T,self.utni.T,self.thetaOLS,self.fac)
        self.do_CS_GMM(self.uti.T,self.thetaOLS,self.fac,alphat.T,alpha,lamda)

        # OLS CS Regressions

        self.LambdasOLS, self.alphasOLS, self.covLamOLS, self.covAlpOLS, = self.CS_OLS(Bni)
        self.tsLamOLS, self.pvLamOLS = self.get_tstat(lamda, self.covLamOLS)
        self.CS_OLS_JS = self.CS_OLS2() [8]
        self.CS_OLS_pval = self.CS_OLS2() [9]

        # OLS correction for AC and CH

        self.covLamOLSC,self.covAlpOLSC = self.correct_ACCH(self.LambdasOLS,self.covLamOLS,
        self.covAlpOLS)

        # GLS CS Regressions

        self.LambdasGLS, self.alphasGLS, self.covLamGLS, self.covAlpGLS = self.CS_GLS(Bni)
        self.tsLamGLS, self.pvLamGLS = self.get_tstat(self.LambdasGLS, self.covLamGLS)
        self.CS_GLS_JS = self.get_Jstat(self.alphasGLS,self.Sigma,corr = self.T*
        (1 + dot(self.LambdasGLS.T,solve(self.vcvfac,self.LambdasGLS)))) [1]

        # GLS correction for AC and CH

        self.covLamGLSC, self.covAlpGLSC = self.correct_ACCH(self.LambdasGLS,
        self.covLamGLS,self.covAlpGLS)

    def TS_OLS(self,const=False):
        """
        This function performs ordinary OLS on the time series. If const = True then an
        intercept is included in regressions.
        """
        if const==False:
            # Get data in SUR shape
            XSUR = zeros([self.N * self.T, self.N * self.k])
            for i in range(self.T):
                XSUR[i * self.N:(i + 1) * self.N, :] = kron(eye(self.N), reshape(self.X[i, :],
                                                                                 (1, self.k)))
        else:
            # Get data in SUR shape
            XSUR = zeros([self.N * self.T, self.N * (self.k + 1)])
            for i in range(self.T):
                XSUR[i * self.N:(i + 1) * self.N, 0:self.N] = eye(self.N)
                XSUR[i * self.N:(i + 1) * self.N, self.N::] = kron(eye(self.N),
                                                                   reshape(self.X[i, :], (1, self.k)))

        # Calculate OLS parameters and residuals

        ylong = reshape(self.y, (self.N * self.T, 1))
        theta = lstsq(XSUR, ylong)[0]
        et = ylong - dot(XSUR, theta)
        sigma = var(et)
        Sigma = sigma * self.IN

        # Reshape residuals to get old shape

        ut = reshape(et, (self.T, self.N))
        ut2 = ut.T
        utm = reshape(mean(ut2, axis=1), (self.N, 1))
        Sigma = dot(ut2 - utm, (ut2 - utm).T) / self.T
        return theta, Sigma, ut

    def CS_OLS(self, beta):
        """
        This function runs a cross-sectional OLS regression
        """

        BB = dot(beta.T, beta)
        if rank(BB, tol=1e-9) < self.k:
            print ("Warning, B.T B has deficient rank of '", rank(BB))
            Binv = pinv(dot(beta.T, beta))
        else:
            Binv = inv(dot(beta.T, beta))
        Lamb = dot(Binv, dot(beta.T, self.ym))
        alphas = self.ym - dot(beta, Lamb)

        # Calculate covariance matrices

        covLamb = dot(Binv, dot(dot(dot(beta.T, self.Sigma), beta), Binv)) / self.T
        t1 = eye(self.N) - dot(beta, dot(Binv, beta.T))
        covAlp = dot(dot(t1, self.Sigma), t1) / self.T

        return Lamb, alphas, covLamb, covAlp

    def CS_OLS2(self):
        """
        For comparison we do cross-sectional OLS with statsmodels package
        """

        beta = self.TSbeta.T
        vcv = dot(self.uti.T, self.uti) / self.T

        # Run CS regression

        model = sm.OLS(self.ym, beta).fit()
        alpha = model.resid
        Lamb = model.params
        vcvLamb = model.cov_params()
        tstat, pvalues = self.get_tstat(Lamb, vcvLamb)
        R2 = model.rsquared
        R2adj = model.rsquared_adj


        # Wald test

        tmp = (self.IN - dot(beta,dot(inv(dot(beta.T,beta)),beta.T)))
        vcvalpha = dot(tmp,dot(vcv,tmp))/self.T
        Jstat = (self.T-self.N-1)*dot(alpha,dot(pinv(vcvalpha),alpha.T))/self.T
        Jpval = 1 - chi2(self.N-self.k).cdf(Jstat)
        self.OLS_pval = Jpval

        return alpha,Lamb,vcvalpha,vcvLamb,tstat,pvalues,R2,R2adj,Jstat,Jpval

    def CS_GLS(self,beta):
        """
        This function runs a cross-sectional GLS regression
        """
        if rank(self.Sigma,tol=1e-9) < self.N:
            BSinv = dot(beta.T,pinv(self.Sigma))
        else:
            BSinv = dot(beta.T,inv(self.Sigma))
        tmp = dot(BSinv,beta)
        if rank(tmp,tol=1e-9) < self.k:
            print ("Warning tmp has deficient rank of ", rank(tmp))
            Binv = pinv(tmp)
        else:
            Binv = inv(tmp)
        Lambdas = dot(Binv,dot(BSinv,self.ym))
        alphas = self.ym - dot(beta,Lambdas)

        # Calculate covariance matrices

        covLam = Binv/self.T
        covAlp = (self.Sigma - dot(beta,dot(Binv,beta.T)))/self.T
        return Lambdas, alphas, covLam, covAlp

    def fmb(self,y,beta):
        """
        This function computes fmb regressions with constant betas
        """
        # Create array to store results

        lambdat = zeros([self.T,self.k])
        alphat = zeros([self.T,self.N])
        avgPort = mean(self.y, axis=0)

        # Run CS regressions

        for i in range(self.T):
            lambdat[i] = lstsq(beta.T,y[i,:])[0]
            alphat[i] = y[i] - dot(lambdat[i].T,beta)

        # Calculate alpha & lambda and their covariance matrices

        alpha = mean(alphat, axis=0)
        lamb = mean(lambdat, axis=0)
        vcvalpha = dot((alphat - alpha).T,(alphat - alpha))/self.T**2
        vcvlamb = dot((lambdat - lamb).T,(lambdat - lamb))/self.T**2

        # Calculate R2, t-statistics and p-values

        R2 = 1 - np.var(alpha) / np.var(avgPort)
        R2adj = R2 - (1-R2) * self.k / (self.N - self.k -1 )
        tstat, pvalue = self.get_tstat(lamb,vcvlamb)

        return alpha,lamb,alphat,lambdat,vcvalpha,vcvlamb,R2,R2adj,tstat,pvalue

    def tv_fmb(self,window):
        """
        This function computes fmb regressions with time-varying betas
        """
        # Create array to store results

        tv_T = self.T - window
        lambdat = zeros([tv_T, self.k])
        alphat = zeros([tv_T, self.N])

        # Run TS & CS regressions

        augFactors = np.hstack((np.ones((len(self.y), 1)), self.X))
        avgPort = mean(self.y, axis=0)
        for i in range(tv_T):
            X_temp = augFactors[i:i + window, :]
            Y_temp = self.y[i:i + window, :]
            out = lstsq(X_temp, Y_temp)
            beta = out[0][1:]
            tv_y = self.y[i, :]
            CS = lstsq(beta.T, tv_y)
            lambdat[i] = CS[0]
            alphat[i] = tv_y - dot(lambdat[i].T, beta)

        # Calculate alpha & lambda and their covariance matrices

        alpha = mean(alphat, axis=0)
        lamb = mean(lambdat, axis=0)
        vcvalpha = dot((alphat - alpha).T, (alphat - alpha)) / tv_T ** 2
        vcvlamb = dot((lambdat - lamb).T, (lambdat - lamb)) / tv_T ** 2

        # Calculate R2, t-statistics and p-values

        R2 = 1 - np.var(alpha) / np.var(avgPort)
        R2adj = R2 - (1 - R2) * self.k / (self.N - self.k - 1)
        tstat, pvalue = self.get_tstat(lamb, vcvlamb)

        return alpha, lamb, alphat, lambdat, vcvalpha, vcvlamb, R2, R2adj, tstat, pvalue

    def fmb_pval(self, alpham, vcvalpha):
        if rank(vcvalpha, tol=1e-9) < self.N:
            self.FMB_JS = dot(alpham.T, dot(pinv(vcvalpha), alpham))

            return chi2.sf(dot(alpham.T, dot(pinv(vcvalpha), alpham)), self.N - self.k)
        else:
            self.FMB_JS = dot(alpham.T, solve(vcvalpha, alpham))
            return chi2.sf(dot(alpham.T, solve(vcvalpha, alpham)), self.N - self.k)

    """
    This function computes GMM functions and finally GMM regressions
    """

    def set_b(self, c, beta, lam=0.0, method='TS'):
        # create parameter vector
        if method == 'TS':
            b = zeros([self.N * (self.k + 1), 1])
        else:
            b = zeros([self.N * (self.k + 1) + self.k, 1])
        b[0:self.N, :] = c
        b[self.N:self.N + self.N * self.k, :] = reshape(beta, (self.N * self.k, 1))
        if method != 'TS':
            b[self.N + self.N * self.k::, :] = reshape(lam, (self.k, 1))
        return b

    def set_g(self, epsilon, f, alpha=0.0, method='TS'):
        if method == 'TS':
            g = zeros([self.N + self.N * self.k, self.T])
        else:
            g = zeros([self.N + self.N * self.k + self.N, self.T])
        for i in range(self.T):
            g[0:self.N, i] = epsilon[:, i]
            g[self.N:(self.N + self.N * self.k), i] = kron(epsilon[:, i], self.fac[:, i])
            if method != 'TS':
                g[(self.N + self.N * self.k)::, i] = alpha[:, i]
        return g

    def set_d(self, lambdas=0.0, B=0.0, method='TS'):
        if method == 'TS':
            d = zeros([self.N + self.N * self.k, self.N + self.N * self.k])
        else:
            d = zeros([self.N * (self.k + 1) + self.N, self.N * (self.k + 1) + self.k])
        d[0:self.N, 0:self.N] = self.IN
        d[0:self.N, self.N:self.N * (self.k + 1)] = kron(self.IN, self.fm.T)
        for i in range(self.N):
            n = self.N + i * self.k
            d[n:(n + self.k), 0:self.N] = kron(self.IN[i, :], self.fm)
            d[n:(n + self.k), self.N:self.N * (self.k + 1)] = kron(self.IN[i, :], self.vcvfac)
        if method != 'TS':
            d[self.N * (self.k + 1)::, self.N:self.N * (self.k + 1)] = kron(self.IN, lambdas.T)
            d[self.N * (self.k + 1)::, self.N * (self.k + 1)::] = B
        return -d

    def set_S(self, m, gt, kernel='HAC'):
        nmom = gt.shape[0]
        S = zeros([nmom, nmom])
        Gamma = zeros([m, nmom, nmom])
        # w = zeros([m,m])
        wd = zeros([m])
        for i in range(m):
            # Get weight matrix
            if kernel == 'HAC':
                wd[i] = 1.0 - i / (m + 1.0)
            elif kernel == "ExpSq":
                wd[i] = exp(-(i + 0.0) ** 2 / (m ** 2 / 4))
            # Get Gammas
            Gamma[i] += dot(gt[:, i::], gt[:, :self.T - i].T) / self.T
            # Get S
            S += wd[i] * (Gamma[i] + Gamma[i].T)
        return S

    def get_varb(self, d, S, a=0.0, method='TS'):
        if method == 'TS':
            dinv = inv(d)
            return dot(dinv, dot(S, dinv.T)) / self.T
        else:
            adinv = inv(dot(a, d))
            varb = dot(adinv, dot(dot(a, S), dot(a.T, adinv.T))) / self.T
            nmom = S.shape[0]
            tmp = eye(nmom) - dot(d, dot(adinv, a))
            vargt = dot(tmp, dot(S, tmp.T)) / self.T
            return varb, vargt

    def do_TS_GMM(self, utu, utr, b, f):
        # Unrestricted case where intercepts are included
        alpha = b[0:self.N]
        gtu = self.set_g(utu, f)
        nmom = gtu.shape[0]
        d = self.set_d()
        m = int(ceil(1.2 * float(self.T) ** (1.0 / 3)))  # int(floor(self.T**(1.0/4.0)))
        Su = self.set_S(m, gtu)
        SIGMAb = self.get_varb(d, Su)

        Sigmaalp = SIGMAb[0:self.N,0:self.N]
        if rank(Sigmaalp,tol=1e-9)<self.N:
            valu = dot(alpha.T,dot(pinv(Sigmaalp),alpha))
        else:
            valu = dot(alpha.T,solve(Sigmaalp,alpha))
        self.TS_GMM_pval_u = squeeze(chi2.sf(valu,self.N - self.k))

        # Restricted case with no intercept included

        gtr = self.set_g(utr,f)
        Sr = self.set_S(m,gtr)
        gTr = reshape(mean(gtr,axis=1),(nmom,1))
        if rank(Sr,tol=1e-9) < nmom:
            valr = self.T*dot(gTr.T,dot(pinv(Sr),gTr))
        else:
            valr = self.T*dot(gTr.T,solve(Sr,gTr))
        self.TS_GMM_pval_r = squeeze(chi2.sf(valr,self.N-self.k))

        # GJ test

        gTu = reshape(mean(gtu,axis=1),(nmom,1))
        val = self.T*(dot(gTr.T,solve(Su,gTr)) - dot(gTu.T,solve(Su,gTu)))
        self.TS_GMM_pval_3 = squeeze(chi2.sf(val,self.N-self.k))

        return

    def do_CS_GMM(self,ut,theta,f,alphat,alpha,lamda):

        # TS intercepts & betas

        c = theta[0:self.N]

        beta = theta[self.N::]
        B = reshape(beta,(self.N,self.k))

        # Setup parameter vector

        b = self.set_b(c,beta,lamda,method='CS')
        npar = b.size

        #Setup moments

        gt = self.set_g(ut,f,alphat,method='CS')
        nmom = gt.shape[0]

        # Setup d matrix

        d = self.set_d(lambdas=lamda,B=B,method='CS')

        #Calculate lag

        m = int(ceil(1.2 * float(self.T)**(1.0/3))) #int(floor(self.T**(1.0/4.0)))

        # Calcuate S matrix
        #S = self.set_S(m,gt) #,kernel="ExpSq"
        #S = S + 1e-8*eye(nmom) #Add jitter since S is near singular

        S = ENS.GMM_cross_sectional(self.y.T,self.X.T) [5]
        self.S = S

        #Setup a matrix

        a = zeros([self.N*(self.k+1) + self.k,self.N*(self.k+1) + self.N])
        a[0:self.N*(self.k+1),0:self.N*(self.k+1)] = eye(self.N*(self.k+1))
        a[self.N*(self.k+1)::,self.N*(self.k+1)::] = B.T
        self.a = a
        self.d = d

        #Calculate variance-covariance matrices

        vcvb, vcvgt= self.get_varb(d,S,a=a,method='CS')
        n = nmom-self.N
        Sigmaalp = vcvgt[n::,n::]
        self.Sigmaalp = Sigmaalp
        vcvLamb = vcvb[-self.k:,-self.k:]
        self.covLamGMM = vcvLamb
        tstats, pvalues = self.get_tstat(lamda,vcvLamb)
        self.tsLamGMM = tstats
        self.pvLamGMM = pvalues
        # Wald test

        if rank(Sigmaalp,tol=1e-9)<self.N:
            val = dot(alpha.T,dot(pinv(Sigmaalp),alpha))
        else:
            val = dot(alpha.T,solve(Sigmaalp,alpha))
        self.CS_GMM_pval1 = squeeze(chi2.sf(val,self.N-self.k))

        # Test of overidentifying restrictions

        gT = reshape(mean(gt,axis=1),(nmom,1))
        self.gT = gT
        if rank(vcvgt,tol=1e-9) < nmom:
            val2 = self.T*dot(gT.T,dot(pinv(S),gT))
        else:
            val2 = self.T*dot(gT.T,solve(S,gT))

        self.CS_GMM_JS = squeeze(val2)
        self.CS_GMM_pval2 = squeeze(chi2.sf(val2,nmom-npar))

        return

    """
    This section computes global functions
    """

    def get_tstat(self, parameter, vcvparameter):
        """
        This function computes t statistic of the regression coefficients
        """
        t_parameter = (parameter.T/np.sqrt(np.diag(vcvparameter))).T
        pvalues = (1-t.cdf(t_parameter, self.N-self.k))*2

        return t_parameter, pvalues

    def get_pseudo_R2(self,vcvport,vcvres):
        """
        Here we get the pseudo R squared statistic
        """
        # Calculate eigenvals and eigenvecs of portfolio covariance matrix

        lam_port, p_port = self.do_spec_dec(vcvport)

        # Calculate eigenvals and eigenvecs of residual covariance matrix

        lam_res, p_res = self.do_spec_dec(vcvres)

        return 1 - sum(lam_res)/sum(lam_port)

    def do_spec_dec(self,A):
        """
        Here we do the spectral decomposition of the square symmetric matrix A
        A is decomposed into A = P V P' where P contains the eigenvectors and V
        is a diagonal matrix containing the eigenvalues of A. The returned eigenvalues
        are ranked in decending order and the ith column of Ps correspondes to the ith
        eigenvalue.
        """
        # Eigen-decomposition of A (this is only for symmetric matrices, for non-
        # symmetric matrices use eig instead of eigh)

        V, P = eigh(A)
        # Calculate indices that sort the eigenvals
        I = argsort(V)
        # Sort eigenvecs
        Ps = P[:,I]
        # Sort eigenvals
        Vs = V[I]
        return Vs, Ps

    def correct_ACCH(self,Lambdas,covLam,covAlp):
        """
        Here we correct for the AC and CH.
        """
        # Calculate the multiplicative correction factor

        if rank(self.vcvfac,tol=1e-9) < self.k:
            print ("Warning vcvfac has deficient rank of ", rank(self.vcvfac))
            mcor = (1 + dot(Lambdas.T,dot(pinv(self.vcvfac),Lambdas)))
        else:
            mcor = (1 + dot(Lambdas.T,solve(self.vcvfac,Lambdas)))

        # Correct the alpha and Lambda

        covLamC = covLam*mcor + self.vcvfac/self.T
        covAlpC = covAlp*mcor

        return covLamC, covAlpC

    def GRS_test(self,alpha,Sigma):
        """
        This test should be used for time series regressions
        """
        if rank(Sigma,tol=1e-9) < self.N:
            print ("Warning Sigma has deficient rank of ", rank(Sigma))
            val = (self.T - self.N - self.k)*dot(alpha.T,dot(pinv(Sigma),alpha))/(self.N*(1 + dot(self.fm.T,solve(self.vcvfac,self.fm))))
        else:
            val = (self.T - self.N - self.k)*dot(alpha.T,solve(Sigma,alpha))/(self.N*(1 + dot(self.fm.T,solve(self.vcvfac,self.fm))))
        return squeeze(f.sf(val,self.N,self.T-self.N-self.k))

    def get_Jstat(self,alpha,sigma,corr = 1.0):
        """
        J test. corr is the prefactor
        """
        if rank(sigma,tol=1e-9) < self.N:
            print ("Warning Sigma has deficient rank of ", rank(sigma))
            val = corr*dot(alpha.T,dot(pinv(sigma),alpha))
        else:
            val = corr*dot(alpha.T,solve(sigma,alpha))
        return chi2.sf(val,self.N-self.k), val

    def CS_OLS_pval_J(self):
        """
        J test. corr is the prefactor
        """
        return self.get_Jstat(self.alphasOLS,self.Sigma,corr = self.T) [0]

    def CCS_OLS_pval_J(self):
        """
        J test. corr is the prefactor
        """
        return self.get_Jstat(self.alphasOLS,self.Sigma,corr = self.T*
        (1 + dot(self.LambdasGLS.T,solve(self.vcvfac,self.LambdasGLS)))) [0]

    def CS_GLS_pval_J(self):
        """
        J test. corr is the prefactor
        """
        return self.get_Jstat(self.alphasGLS,self.Sigma,corr = self.T) [0]

    def CCS_GLS_pval_J(self):
        """
        J test. corr is the prefactor
        """
        return self.get_Jstat(self.alphasGLS,self.Sigma,corr = self.T*
        (1 + dot(self.LambdasGLS.T,solve(self.vcvfac,self.LambdasGLS)))) [0]
