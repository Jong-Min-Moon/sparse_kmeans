%% test_variable_selection_clean_spectral
% @export
function test_variable_selection_clean_spectral()
%TEST_ISEE_VARIABLE_SELECTION_VS_FLIP
%   Evaluates variable selection robustness to clustering error at flip ratios 0.1, 0.2, 0.3
    rng(1);
    % Parameters
    p = 800;
    n = 200;
    s = 10;
    rho = 0.5;
 
    % Generate true precision matrix (tridiagonal)
    [X, label_true, mu1, mu2, sep, ~, beta_star]  = generate_gaussian_data(n, p, 4, 'ER', 1, 1/2);
    % Selection threshold
    threshold = sqrt(log(p) * log(n) / n);
    fprintf('Selection threshold: %.4f\n\n', threshold);
    % Header
    fprintf('%10s  %5s  %5s  %5s  %6s  %6s\n', 'FlipRatio', 'TP', 'FN', 'FP', 'TPR', 'FPR');
    fprintf('%s\n', repmat('-', 1, 40));
    % Loop over flip ratios
            % Perturb cluster labels
            cluster_est = cluster_spectral(X', 2);
            % Run estimator
            [mean_vec, ~, ~, ~] = ISEE_bicluster_parallel(X', cluster_est);
            selected = select_variable_ISEE_clean(mean_vec, n);
            TP = sum(selected(1:s));
            FN = s - TP;
            FP = sum(selected(s+1:end));
     
        % Report
        fprintf('  %5.2f  %5.2f  \n', ...
             TP , FN, FP);
    fprintf('\n✓ Full variable selection robustness evaluation completed.\n');
end
