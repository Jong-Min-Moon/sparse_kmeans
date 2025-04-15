function [intercept, residual] = get_intercept_residual_lasso(response, predictor)
%% get_intercept_residual_lasso
% @export
% 
% Fits a lasso model and returns intercept and residual
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
