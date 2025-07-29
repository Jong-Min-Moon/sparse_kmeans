%% get_intercept_residual_lasso
% @export
% 
% Computes the intercept and residuals from a Lasso-penalized linear regression. 
% Given a response vector and a predictor matrix, the predictor matrix is automatically 
% standardized before fitting. This function fits a Lasso with many values of 
% $\lambda$, selects the model with the lowest AIC, extracts the intercept and 
% slope coefficients, and returns the residuals.
% 
% 
% 
% INPUTS:
%% 
% * response  - An n-by-1 vector of response values.
% * predictor - An n-by-p matrix of predictor variables.
%% 
% OUTPUTS:
%% 
% * Intercept - The scalar intercept term from the selected Lasso model.
% * residual  - An n-by-1 vector of residuals from the fitted model.
function [intercept, residual] = get_intercept_residual_lasso(response, predictor)                 
  
model_lasso = glm_gaussian(response, predictor); 
    
    fit = penalized(model_lasso, @p_lasso, "standardize", true); % Fit lasso
    % Select model with minimum AIC
    AIC = goodness_of_fit('aic', fit);
    [~, min_aic_idx] = min(AIC);
    beta = fit.beta(:,min_aic_idx);
    % Extract intercept and slope
    intercept = beta(1);
    slope = beta(2:end);
    % Compute residual
    residual = response - intercept - predictor * slope;
end
%% 
% 
