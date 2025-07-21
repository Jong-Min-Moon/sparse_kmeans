function test_ISEE_variable_selection_robustness()
%% test_ISEE_bicluster_parallel
% @export
%TEST_ISEE_VARIABLE_SELECTION_ROBUSTNESS
%   Tests variable selection power (TP/FN) under clustering perturbation using Omega * mean_diff
    rng(42);
    % Parameters
    p = 100;
    n = 200;
    s = 10;
    rho = 0.5;
    n_trials = 10;
    flip_ratio = 0.1;
    % Generate sparse precision (tridiagonal)
    Omega_true = diag(ones(p,1));
    Omega_true = Omega_true + diag(rho * ones(p-1,1), 1) + diag(rho * ones(p-1,1), -1);
    Sigma_true = inv(Omega_true);
    % True cluster means
    mu1 = zeros(p,1); mu2 = zeros(p,1);
    mu1(1:s) = 1; mu2(1:s) = -1;
    % Generate data once
    n1 = n/2; n2 = n - n1;
    true_cluster = [ones(1, n1), 2 * ones(1, n2)];
    X = zeros(p, n);
    X(:,1:n1) = mvnrnd(mu1, Sigma_true, n1)';
    X(:,n1+1:end) = mvnrnd(mu2, Sigma_true, n2)';
    % Set threshold
    threshold = sqrt(log(p) * log(n) / n);
    fprintf('Selection threshold: %.4f\n', threshold);
    % Results
    TPs = zeros(n_trials,1);
    FNs = zeros(n_trials,1);
    for t = 1:n_trials
        % Perturb cluster labels
        cluster_est = true_cluster;
        flip_idx = randperm(n, round(flip_ratio * n));
        cluster_est(flip_idx) = 3 - cluster_est(flip_idx);  % flip 1 <-> 2
        % Run estimator
        [mean_vec, ~, ~, ~] = ISEE_bicluster_parallel(X, cluster_est);
        % Compute estimated beta
        mu_diff_hat = mean_vec(:,1) - mean_vec(:,2);
        beta_hat = Omega_true * mu_diff_hat;
        % Evaluate
        selected = abs(beta_hat) > threshold;
        TP = sum(selected(1:s));
        FN = s - TP;
        TPs(t) = TP;
        FNs(t) = FN;
        fprintf('[Trial %d] TP = %d/%d, FN = %d\n', t, TP, s, FN);
    end
    % Summary
    avg_TP = mean(TPs);
    avg_FN = mean(FNs);
    fprintf('\nAverage TP over %d trials: %.2f/%d\n', n_trials, avg_TP, s);
    fprintf('Average FN over %d trials: %.2f\n', n_trials, avg_FN);
    assert(avg_TP >= 8, 'Too many missed signals on average (low TP)');
    fprintf('✓ Variable selection power robust under cluster perturbation.\n');
end
