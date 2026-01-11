function test_ISEE_bicluster_variable_selection()
%% test_ISEE_bicluster_parallel
% @export
%TEST_ISEE_BICLUSTER_VARIABLE_SELECTION
%   Check variable selection from Omega*(mu1 - mu2) under perturbed clustering
    rng(999);
    % Setup
    p = 100;
    n = 200;
    s = 10;
    rho = 0.5;
    % Precision: sparse tridiagonal
    Omega_true = diag(ones(p,1));
    Omega_true = Omega_true + diag(rho * ones(p-1,1), 1) + diag(rho * ones(p-1,1), -1);
    Sigma_true = inv(Omega_true);
    % True sparse means
    mu1 = zeros(p,1); mu2 = zeros(p,1);
    mu1(1:s) = 1; mu2(1:s) = -1;
    % True cluster labels
    n1 = n/2; n2 = n - n1;
    true_cluster = [ones(1, n1), 2 * ones(1, n2)];
    % Simulate data
    X = zeros(p, n);
    X(:, 1:n1) = mvnrnd(mu1, Sigma_true, n1)';
    X(:, n1+1:end) = mvnrnd(mu2, Sigma_true, n2)';
    % Perturb cluster labels
    cluster_est = true_cluster;
    flip_idx = randperm(n, round(0.1 * n));
    cluster_est(flip_idx) = 3 - cluster_est(flip_idx);
    % Run ISEE-based biclustering
    [mean_vec, ~, ~, ~] = ISEE_bicluster_parallel(X, cluster_est);
    % Discriminant vector estimate
    mu_diff = mean_vec(:,1) - mean_vec(:,2);
    beta_hat = Omega_true * mu_diff;
    % Variable selection threshold
    threshold = sqrt(log(p) * log(n) / n);
    selected = abs(beta_hat) > threshold;
    % Evaluate selection
    true_support = false(p,1); true_support(1:s) = true;
    TP = sum(selected(1:s));
    FP = sum(selected(s+1:end));
    FN = s - TP; TN = p - s - FP;
    fprintf('Threshold = %.4f\n', threshold);
    fprintf('TP = %d/%d, FP = %d, FN = %d, TN = %d\n', TP, s, FP, FN, TN);
    fprintf('TPR = %.2f, FPR = %.2f\n', TP/s, FP/(p-s));
    assert(TP/s > 0.8, 'True signals not well recovered.');
    assert(FP/(p-s) < 0.2, 'Too many false positives.');
    fprintf('✓ Test passed: sparsity in Omega * mean_vec recovered.\n');
end
