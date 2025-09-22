%% fit_elasticNet
% @export
function [bestBeta, bestIntercept, bestAlpha, bestMSE] = fit_elasticNet(X, y)
%TUNELASSO Fit LASSO models over a grid of alpha values and select best by CV MSE
%
%   [bestBeta, bestIntercept, bestAlpha, bestMSE] = tuneLasso(X, y)
%
%   Inputs:
%       X - n x p design matrix
%       y - n x 1 response vector
%
%   Outputs:
%       bestBeta     - best coefficient vector
%       bestIntercept- intercept corresponding to best fit
    alphas = 0.1:0.1:1;   % gamma/alpha candidates
    bestMSE = Inf;
    bestAlpha = NaN;
    bestBeta = [];
    bestIntercept = NaN;
    for a = alphas
        [B, FitInfo] = lasso(X, y, ...
            'CV', 10, ...
            'Alpha', a, ...
            'Intercept', true, ...
            'Standardize', true);
        mseVal = FitInfo.MSE(FitInfo.IndexMinMSE);
        if mseVal < bestMSE
            bestMSE = mseVal;
            bestAlpha = a;
            bestBeta = B(:, FitInfo.IndexMinMSE)';
            bestIntercept = FitInfo.Intercept(FitInfo.IndexMinMSE);
        end
    end
end
%% 
% 
% 
% 
