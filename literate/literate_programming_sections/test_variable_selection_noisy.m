function test_variable_selection_noisy()
%% test_variable_selection_noisy
% @export
%TEST_ISEE_VARIABLE_SELECTION_VS_FLIP
%   Evaluates variable selection robustness to clustering error at flip ratios 0.1, 0.2, 0.3
    rng(1);
    % Parameters
    p = 800;
    n = 200;
    s = 10;
    rho = 0.5;
    n_trials = 3;
    flip_ratios = [0.1, 0.2, 0.3];
    % Generate true precision matrix (tridiagonal)
    [X, label_true, mu1, mu2, sep, ~, beta_star]  = generate_gaussian_data(n, p, 4, 'ER', 1, 1/2);
    % Selection threshold
    threshold = sqrt(log(p) * log(n) / n);
    fprintf('Selection threshold: %.4f\n\n', threshold);
    % Header
    fprintf('%10s  %5s  %5s  %5s  %6s  %6s\n', 'FlipRatio', 'TP', 'FN', 'FP', 'TPR', 'FPR');
    fprintf('%s\n', repmat('-', 1, 40));
    % Loop over flip ratios
    for flip_ratio = flip_ratios
        TPs = zeros(n_trials, 1);
        FNs = zeros(n_trials, 1);
        FPs = zeros(n_trials, 1);
        for t = 1:n_trials
            % Perturb cluster labels
            cluster_estimate = label_true';
            flip_idx = randperm(n, round(flip_ratio * n));
            cluster_estimate(flip_idx) = 3 - cluster_estimate(flip_idx);
            % Run estimator
            [mean_vec, noise_mat, Omega_diag_hat, mean_mat] = ISEE_bicluster_parallel(X', cluster_estimate);
            selected = select_variable_ISEE_noisy(mean_mat, noise_mat, Omega_diag_hat, cluster_estimate);
            TP = sum(selected(1:s));
            FN = s - TP;
            FP = sum(selected(s+1:end));
            TPs(t) = TP;
            FNs(t) = FN;
            FPs(t) = FP;
        end
        % Aggregate
        avg_TP = mean(TPs);
        avg_FN = mean(FNs);
        avg_FP = mean(FPs);
        TPR = avg_TP / s;
        FPR = avg_FP / (p - s);
        % Report
        fprintf('%10.1f  %5.2f  %5.2f  %5.2f  %6.2f  %6.2f\n', ...
            flip_ratio, avg_TP, avg_FN, avg_FP, TPR, FPR);
    end
    fprintf('\n✓ Full variable selection robustness evaluation completed.\n');
end
