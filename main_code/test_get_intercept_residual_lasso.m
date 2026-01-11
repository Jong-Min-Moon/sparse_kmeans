%% test_get_intercept_residual_lasso
% @export
function test_get_intercept_residual_lasso()
%TEST_GET_INTERCEPT_RESIDUAL_LASSO Verifies Lasso estimates active and zero coefficients correctly,
% and ensures output variables have correct sizes.
    rng(42);  % For reproducibility
    n = 100;  % Number of observations
    p = 10;   % Number of predictors
    % True coefficients: sparse (only first 3 non-zero)
    true_intercept = 1.5;
    true_slope = [3; -2; 1.5; zeros(p - 3, 1)];
    % Simulate predictor matrix and response vector
    X = randn(n, p);
    noise = randn(n, 1) * 0.5;
    y = true_intercept + X * true_slope + noise;
    % Run Lasso-based regression
    [intercept_est, residual] = get_intercept_residual_lasso(y, X);
    % --- Check output sizes ---
    assert(isscalar(intercept_est), 'Intercept should be a scalar.');
    assert(isequal(size(residual), [n, 1]), 'Residual must be an n-by-1 column vector.');
    % Estimate slope using residual (for evaluation only)
    slope_est = (y - intercept_est - residual)' * X / (X' * X);  % 1 x p row vector
    % --- Evaluate recovery of active coefficients ---
    active_true = true_slope(1:3);
    active_est = slope_est(1:3);
    mse_active = mean((active_est - active_true').^2);
    fprintf('MSE on active coefficients: %.4f\n', mse_active);
    assert(mse_active < 0.5, 'Active coefficients are not well estimated.');
    % --- Evaluate shrinkage of inactive coefficients ---
    inactive_est = slope_est(4:end);
    max_inactive = max(abs(inactive_est));
    fprintf('Max abs value on inactive coefficients: %.4e\n', max_inactive);
    assert(max_inactive < 0.1, 'Inactive coefficients are not shrunk to zero.');
    % --- Additional diagnostics ---
    fprintf('Estimated intercept: %.4f (true = %.4f)\n', intercept_est, true_intercept);
    fprintf('Residual variance: %.4f\n', var(residual));
end
