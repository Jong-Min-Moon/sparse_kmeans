% Test Merged SDP KMeans Bandit

% Check Architecture
fprintf('MATLAB Architecture: %s\n', computer('arch'));

% Add dependencies
addpath(genpath('/Users/jmmoon/Documents/GitHub/sparse_kmeans/SDPNAL+v1.0'));

% 1. Generate Synthetic Data
p = 10;
n_per_cluster = 20;
n = 2 * n_per_cluster;
K = 2;

mu1 = zeros(p, 1);
mu2 = zeros(p, 1);
mu2(1:3) = 5; 
% Signal in first 3 features

X1 = mu1 + randn(p, n_per_cluster);
X2 = mu2 + randn(p, n_per_cluster);
X = [X1, X2];

cluster_true = [ones(1, n_per_cluster), 2*ones(1, n_per_cluster)];

% 2. Instantiate Bandit
C = 0.5;
bandit = sdp_kmeans_bandit(X, K, C);

% 3. Test fit_predict with cluster_true (Simulation mode)
fprintf('Testing fit_predict WITH cluster_true...\n');
n_iter = 5;
bandit.fit_predict(n_iter, cluster_true);

% Check accuracy dictionary
if ~isempty(bandit.acc_dict)
    fprintf('Accuracy dictionary populated. Values:\n');
    disp(values(bandit.acc_dict));
else
    error('Accuracy dictionary NOT populated!');
end

% 4. Test evaluate_discovery
support = [1, 2, 3];
[tp, fp, fn, survived] = bandit.evaluate_discovery(support);
fprintf('Discovery metrics:\n');
fprintf('TP: %s\n', mat2str(tp'));
fprintf('FP: %s\n', mat2str(fp'));

% 5. Test save_results
output_dir = 'test_results';
rep = 1;
Delta = 5;
bandit.save_results(output_dir, rep, Delta, support);

% 6. Test fit_predict WITHOUT cluster_true (Normal mode)
fprintf('\nTesting fit_predict WITHOUT cluster_true...\n');
bandit_normal = sdp_kmeans_bandit(X, K, C);
% New instance

bandit_normal.fit_predict(n_iter);

fprintf('Test Complete due to no errors.\n');
