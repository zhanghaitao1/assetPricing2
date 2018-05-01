%https://cn.mathworks.com/matlabcentral/fileexchange/45898-grs-test-statistic


function[FGRS, pGRS] = fGRS(alpha, eps, mu)
%   [FGRS, pGRS] = fGRS(alpha, eps, mu) returns the GRS test statistic
%       and its corresponding p-Value proposed in 
%       Gibbons, Ross, Shanken (1989), to test the hypothesis:
%       alpha1 = alpha2 = ... = alphaN = 0. That is if the alphas from a
%       time series Regression on N Test Assets are cummulativly zero.
%       Used e.g. in Artmann/Finter/Kempf/Koch/Theissen (2012), p. 33ff.
%
%	parameters: 
%		N is the number of assets (or the number of regressions)
%		T is the length of the time series for every assets
%		L is the number of factors
%
%   Input:
%       alpha       := Nx1 Vector of intercepts from TS-Rergression
%       eps         := TxN Matrix of Residuals from TS-Regression
%       mu          := TxL Matrix of Excess Factor Returns
%
%   Output:
%       FGRS        := 1x1 Scalar of GRS Test-Value
%       pGRS        := 1x1 Scalar of P-Value from an F-Distribution.
%
%%% Sven Thies 24/02/2014 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[T,N] = size(eps);
L = size(mu,2);

% Mean Excess Factor Returns
mu_mean = mean(mu)';

% Compute Covariance Matrix of eps
COV_e = eps'*eps / (T-L-1);

% Compute Covariance Matrix of Factor Returns
COV_f = ((mu - repmat(mu_mean',T,1))' * (mu - repmat(mu_mean',T,1))) / (T-1);

% Compute GRS statistic
FGRS = (T/N) * ((T-N-L)/(T-L-1)) * ((alpha' * inv(COV_e) * alpha) / (1 + (mu_mean' * inv(COV_f) * mu_mean))); 

% p-Value for GRS statistic: GRS ~ F(N,T-N-L)
pGRS = 1-fcdf(GRS, N, (T-N-L));

end