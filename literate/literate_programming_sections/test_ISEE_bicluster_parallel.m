function test_ISEE_bicluster_parallel()
%% test_ISEE_bicluster_parallel
% @export
%TEST_ISEE_VARIABLE_SELECTION_VS_FLIP
%   Evaluates variable selection robustness to clustering error at flip ratios 0.1, 0.2, 0.3
    rng(1);
    % Parameters
    p = 1000;
    n = 200;
    s = 10;
    rho = 0.5;
    n_trials = 10;
    flip_ratios = [0.1, 0.2, 0.3];
    % Generate true precision matrix (tridiagonal)
    Omega_true = diag(ones(p,1));
    Omega_true = Omega_true + diag(rho * ones(p-1,1), 1) + diag(rho * ones(p-1,1), -1);
    Sigma_true = inv(Omega_true);
    % True sparse means
    mu1 = zeros(p,1); mu2 = zeros(p,1);
    mu1(1:s) = 1; mu2(1:s) = -1;
    delta_mu = mu1 - mu2;
    mahalanobis_dist = sqrt(delta_mu' * Omega_true * delta_mu);
    fprintf('Mahalanobis distance between mu1 and mu2: %.4f\n\n', mahalanobis_dist);
    % Generate fixed data
    n1 = n/2; n2 = n - n1;
    true_cluster = [ones(1, n1), 2 * ones(1, n2)];
    X = zeros(p, n);
    X(:, 1:n1) = mvnrnd(mu1, Sigma_true, n1)';
    X(:, n1+1:end) = mvnrnd(mu2, Sigma_true, n2)';
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
            cluster_est = true_cluster;
            flip_idx = randperm(n, round(flip_ratio * n));
            cluster_est(flip_idx) = 3 - cluster_est(flip_idx);
            % Run estimator
            [mean_vec, ~, ~, ~] = ISEE_bicluster_parallel(X, cluster_est);
            % Compute beta_hat = Omega * (mu1 - mu2)
            mu_diff_hat = mean_vec(:,1) - mean_vec(:,2);
            beta_hat = Omega_true * mu_diff_hat;
            selected = abs(beta_hat) > threshold;
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
