function test_ISEE_variable_selection_vs_flip()
%% test_ISEE_bicluster_parallel
% @export
%TEST_ISEE_VARIABLE_SELECTION_VS_FLIP
%   Evaluates variable selection robustness to clustering error at flip ratios 0.1, 0.2, 0.3
    rng(123);
    % Parameters
    p = 100;
    n = 200;
    s = 10;
    rho = 0.5;
    n_trials = 10;
    flip_ratios = [0.1, 0.2, 0.3];
    % Generate true precision
    Omega_true = diag(ones(p,1));
    Omega_true = Omega_true + diag(rho * ones(p-1,1), 1) + diag(rho * ones(p-1,1), -1);
    Sigma_true = inv(Omega_true);
    % True sparse means
    mu1 = zeros(p,1); mu2 = zeros(p,1);
    mu1(1:s) = 1; mu2(1:s) = -1;
    % Generate data once
    n1 = n/2; n2 = n - n1;
    true_cluster = [ones(1, n1), 2 * ones(1, n2)];
    X = zeros(p, n);
    X(:, 1:n1) = mvnrnd(mu1, Sigma_true, n1)';
    X(:, n1+1:end) = mvnrnd(mu2, Sigma_true, n2)';
    threshold = sqrt(log(p) * log(n) / n);
    fprintf('Threshold for selection: %.4f\n\n', threshold);
    % Loop over flip ratios
    for flip_ratio = flip_ratios
        TPs = zeros(n_trials, 1);
        FNs = zeros(n_trials, 1);
        for t = 1:n_trials
            % Perturb labels
            cluster_est = true_cluster;
            flip_idx = randperm(n, round(flip_ratio * n));
            cluster_est(flip_idx) = 3 - cluster_est(flip_idx);
            % Run ISEE estimator
            [mean_vec, ~, ~, ~] = ISEE_bicluster_parallel(X, cluster_est);
            % Estimate beta
            mu_diff_hat = mean_vec(:,1) - mean_vec(:,2);
            beta_hat = Omega_true * mu_diff_hat;
            selected = abs(beta_hat) > threshold;
            TP = sum(selected(1:s));
            FN = s - TP;
            TPs(t) = TP;
            FNs(t) = FN;
        end
        % Summary stats
        avg_TP = mean(TPs);
        avg_FN = mean(FNs);
        fprintf('Flip ratio = %.1f → Avg TP = %.2f / %d, Avg FN = %.2f\n', ...
            flip_ratio, avg_TP, s, avg_FN);
    end
    fprintf('\n✓ Flip robustness test completed across flip ratios.\n');
end
