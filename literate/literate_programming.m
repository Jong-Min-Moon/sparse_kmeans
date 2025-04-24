function z = dummy(x,y)
%% DUMMY 
    z = x+y;
end
%% 
% 
%% 
% 
%% Basic functions
%% 
%% kmeans_sdp_pengwei
% @export
% 
% The following implementation, originally written by Mixon, Villar, and Ward 
% and last edited on January 20, 2024, uses SDPNAL+ [3] to solve the Peng and 
% Wei k-means SDP formulation [2], following the approach described in [1]. The 
% original version accepts a $p\times n$ data matrix as input. To accommodate 
% both isotropic and non-isotropic cases, we modify the code to accept an affinity 
% matrix instead. Given an affinity matrix $A$ , The code solves the following 
% problem:
% 
% $\hat{Z} = \arg\max_{Z \in \mathbb{R}^{n \times n}} \langle A, Z \rangle \quad 
% \text{subject to} \quad Z \succeq 0,\; \mathrm{tr}(Z) = K,\; Z \mathbf{1}_n 
% = \mathbf{1}_n,\; Z \geq 0$.
% 
% Inputs:
%% 
% * A : A n x n array of affinity matrix where n denotes the number of observations. 
% * k: The number of clusters.
% * Outputs:
% * X:  A N x N array corresponding to the solution of Peng and Wei's solution.            
%% 
% References:
%% 
% # Mixon, Villar, Ward. Clustering subgaussian mixtures via semidefinite programming
% # Peng, Wei. Approximating k-means-type clustering via semidefinite programming
% # Yang, Sun, Toh. Sdpnal+: a majorized semismooth newton-cg augmented lagrangian 
% method for semidefinite programming with nonnegative constraints
function X=kmeans_sdp_pengwei(A, k)
D = -A;
N=size(A,2);
% SDP definition for SDPNAL+
n=N;
C{1}=D;
blk{1,1}='s'; blk{1,2}=n;
b=zeros(n+1,1);
Auxt=spalloc(n*(n+1)/2, n+1, 5*n);
Auxt(:,1)=svec(blk(1,:), eye(n),1);
b(1,1)=k;
idx=2;
for i=1:n
    A=zeros(n,n);
    A(:,i)=ones(n,1);
    A(i,:)=A(i,:)+ones(1,n);
    b(idx,1)=2;
    Auxt(:,idx)= svec(blk(1,:), A,1);
    idx=idx+1;
end
At{1}=sparse(Auxt);
OPTIONS.maxiter = 50000;
OPTIONS.tol = 1e-6;
OPTIONS.printlevel = 0;
% SDPNAL+ call
[obj,X,s,y,S,Z,y2,v,info,runhist]=sdpnalplus(blk,At,C,b,0,[],[],[],[],OPTIONS);
X=cell2mat(X);
end
%% Implementing our iterative algorithm
% Our iterative algorithm has the following structure:
%% 
% * Initialization
% * for loop
% * one iteration
% * variable selection        
% * SDP clustering
% * stopping rule
%% 
% We implement these step by step.
%% Initialization
% For now, we only implement the spectral clustering.
% 
% 
%% cluster_spectral
% @export
% 
% Inputs
%% 
% * x: $p\times n$data matrix, where $p$ is the data dimension and $n$ is the 
% sample size
% * k: positive integer. number of cluster.
%% 
% Outputs
%% 
% * cluster_est: n array of positive integers, where n is the sample size. ex. 
% [1 2 1 2 3 4 2 ]
function cluster_est = cluster_spectral(x, k)
    n = size(x,2);
    H_hat = (x' * x)/n; %compute affinity matrix
    [V,D] = eig(H_hat);
    [d,ind] = sort(diag(D), "descend");
    Ds = D(ind,ind);
    Vs = V(:,ind);
    [cluster_est,C] = kmeans(Vs(:,1),k);
    cluster_est= cluster_est';
end
%% 
% 
% 
% We begin by implementing a single step of the algorithm, which we then use 
% to construct the full iterative procedure. Each step consists of two components: 
% variable selection and SDP-based clustering. We implement these two parts sequentially 
% and combine them into a single step function.
%% Variable selection
% Variable selection is two-step:
%% 
% # covariance structure estimation 
%% 
% 
%% ISEE_bicluster
% @export
% 
% ISEE performs numerous Lasso regressions. We provide two versions: a plain 
% version (this function) suitable for running on a local machine, and a parallel 
% version (|ISEE_bicluster_parallel|) optimized for faster execution on computing 
% clusters.
%% 
% Inputs
%% 
% * x: $p\times n$ data matrix, where $p$ is the data dimension and $n$ is the 
% sample size
% * cluster_est_now: $n$ array of positive integers, where n is the sample size. 
% ex. [1 2 1 2 3 4 2 ]
%% 
% Outputs
%% 
% * mean_now: $p\times n$ matrix of  cluster center part of the innovated data 
% matrix (pre-multiplied by precision matrix), where $p$ is the data dimension 
% and $n$ is the sample size
% * noise_now: $p\times n$ matrix of Gaussian noise part of the data matrix 
% (pre-multiplied by precision matrix), where $p$ is the data dimension and $n$ 
% is the sample size
% * Omega_diag_hat: $p$ vector of diagonal entries of precision matrix 
%% 
% 
function [mean_vec, noise_mat, Omega_diag_hat, mean_mat] = ISEE_bicluster(x, cluster_est_now)
%ISEE_BICLUSTER_PARALLEL Estimates means and noise using blockwise Lasso regressions.
% 
% INPUT:
%   x               - p × n data matrix
%   cluster_est_now - 1 × n vector of cluster labels (must be 1 or 2)
%
% OUTPUT:
%   mean_vec        - p × 2 matrix; each column is the estimated mean vector for one cluster
%   noise_mat       - p × n matrix of estimated noise values
%   Omega_diag_hat  - p × 1 vector of estimated diagonals of precision matrix
%   mean_mat        - p × n matrix of cluster-wise sample means
    [p, n] = size(x);
    n_regression = floor(p / 2);
    mean_vec = zeros(p, 2);
    noise_mat = zeros(p, n);
    Omega_diag_hat = zeros(p, 1);
    % Preallocate output pieces for parfor
    mean_vec_parts = cell(n_regression, 1);
    noise_mat_parts = cell(n_regression, 1);
    Omega_diag_parts = cell(n_regression, 1);
    for i = 1:n_regression
        rows_idx = [2*i - 1, 2*i];
        predictors_idx = true(1, p);
        predictors_idx(rows_idx) = false;
        E_Al = zeros(2, n);
        alpha_Al = zeros(2, 2);
        for c = 1:2
            cluster_mask = (cluster_est_now == c);
            x_cluster = x(:, cluster_mask);
            predictor_now = x_cluster(predictors_idx, :)';
            for j = 1:2
                row_idx = rows_idx(j);
                response_now = x_cluster(row_idx, :)';
                [intercept, residual] = get_intercept_residual_lasso(response_now, predictor_now);
                E_Al(j, cluster_mask) = residual;
                alpha_Al(j, c) = intercept;
            end
        end
        Omega_hat_Al = inv(E_Al * E_Al') * n;
        % Store only the 2-row results
        mean_local = Omega_hat_Al * alpha_Al;  % 2 × 2
        noise_local = Omega_hat_Al * E_Al;     % 2 × n
        diag_local = diag(Omega_hat_Al);       % 2 × 1
        % Store using structs
        mean_vec_parts{i} = struct('idx', rows_idx, 'val', mean_local);
        noise_mat_parts{i} = struct('idx', rows_idx, 'val', noise_local);
        Omega_diag_parts{i} = struct('idx', rows_idx, 'val', diag_local);
    end
    % Aggregate results after parfor
    for i = 1:n_regression
        idx = mean_vec_parts{i}.idx;
        mean_vec(idx, :) = mean_vec_parts{i}.val;
        noise_mat(idx, :) = noise_mat_parts{i}.val;
        Omega_diag_hat(idx) = Omega_diag_parts{i}.val;
    end
    % Construct sample-wise mean matrix
    mean_mat = zeros(p, n);
    for c = 1:2
        cluster_mask = (cluster_est_now == c);
        mean_mat(:, cluster_mask) = repmat(mean_vec(:, c), 1, sum(cluster_mask));
    end
end
%% 
% 
% 
% 
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
%% ISEE_bicluster_parallel
% @export
% 
% Inputs
%% 
% * x: $p\times n$ data matrix, where $p$ is the data dimension and $n$ is the 
% sample size
% * cluster_est_now: $n$ array of positive integers, where n is the sample size. 
% ex. [1 2 1 2 3 4 2 ]
%% 
% Outputs
%% 
% * mean_now: $p\times n$ matrix of  cluster center part of the innovated data 
% matrix (pre-multiplied by precision matrix), where $p$ is the data dimension 
% and $n$ is the sample size
% * noise_now: $p\times n$ matrix of Gaussian noise part of the data matrix 
% (pre-multiplied by precision matrix), where $p$ is the data dimension and $n$ 
% is the sample size
% * Omega_diag_hat: $p$ vector of diagonal entries of precision matrix 
%% 
% 
function [mean_vec, noise_mat, Omega_diag_hat, mean_mat] = ISEE_bicluster_parallel(x, cluster_est_now)
%ISEE_BICLUSTER_PARALLEL Estimates means and noise using blockwise Lasso regressions.
% 
% INPUT:
%   x               - p × n data matrix
%   cluster_est_now - 1 × n vector of cluster labels (must be 1 or 2)
%
% OUTPUT:
%   mean_vec        - p × 2 matrix; each column is the estimated mean vector for one cluster
%   noise_mat       - p × n matrix of estimated noise values
%   Omega_diag_hat  - p × 1 vector of estimated diagonals of precision matrix
%   mean_mat        - p × n matrix of cluster-wise sample means
    [p, n] = size(x);
    n_regression = floor(p / 2);
    mean_vec = zeros(p, 2);
    noise_mat = zeros(p, n);
    Omega_diag_hat = zeros(p, 1);
    % Preallocate output pieces for parfor
    mean_vec_parts = cell(n_regression, 1);
    noise_mat_parts = cell(n_regression, 1);
    Omega_diag_parts = cell(n_regression, 1);
    parfor i = 1:n_regression
        rows_idx = [2*i - 1, 2*i];
        predictors_idx = true(1, p);
        predictors_idx(rows_idx) = false;
        E_Al = zeros(2, n);
        alpha_Al = zeros(2, 2);
        for c = 1:2
            cluster_mask = (cluster_est_now == c);
            x_cluster = x(:, cluster_mask);
            predictor_now = x_cluster(predictors_idx, :)';
            for j = 1:2
                row_idx = rows_idx(j);
                response_now = x_cluster(row_idx, :)';
                [intercept, residual] = get_intercept_residual_lasso(response_now, predictor_now);
                E_Al(j, cluster_mask) = residual;
                alpha_Al(j, c) = intercept;
            end
        end
        Omega_hat_Al = inv(E_Al * E_Al') * n;
        % Store only the 2-row results
        mean_local = Omega_hat_Al * alpha_Al;  % 2 × 2
        noise_local = Omega_hat_Al * E_Al;     % 2 × n
        diag_local = diag(Omega_hat_Al);       % 2 × 1
        % Store using structs
        mean_vec_parts{i} = struct('idx', rows_idx, 'val', mean_local);
        noise_mat_parts{i} = struct('idx', rows_idx, 'val', noise_local);
        Omega_diag_parts{i} = struct('idx', rows_idx, 'val', diag_local);
    end
    % Aggregate results after parfor
    for i = 1:n_regression
        idx = mean_vec_parts{i}.idx;
        mean_vec(idx, :) = mean_vec_parts{i}.val;
        noise_mat(idx, :) = noise_mat_parts{i}.val;
        Omega_diag_hat(idx) = Omega_diag_parts{i}.val;
    end
    % Construct sample-wise mean matrix
    mean_mat = zeros(p, n);
    for c = 1:2
        cluster_mask = (cluster_est_now == c);
        mean_mat(:, cluster_mask) = repmat(mean_vec(:, c), 1, sum(cluster_mask));
    end
end
%% 
% 
% 
% 
%% 
%% select_variable_ISEE_noisy
% @export
% 
% Inputs:
%% 
% * mean_now: $p\times n$ matrix of  cluster center part of the innovated data 
% matrix (pre-multiplied by precision matrix), where $p$ is the data dimension 
% and $n$ is the sample size
% * noise_now: $p\times n$ matrix of Gaussian noise part of the data matrix 
% (pre-multiplied by precision matrix), where $p$ is the data dimension and $n$ 
% is the sample size
% * Omega_diag_hat: $p$ vector of diagonal entries of precision matrix
% * cluster_est_prev: $n$ array of positive integers, where n is the sample 
% size. ex. [1 2 1 2 3 4 2 ]
%% 
% Outputs:
%% 
% * s_hat: $p$ boolean vector, where true indicates that variable is selected
function s_hat = select_variable_ISEE_noisy(mean_now, noise_now, Omega_diag_hat, cluster_est_prev)
    x_tilde_now = mean_now + noise_now;
    p = size(mean_now,1);
    n = size(mean_now,2);
    thres = sqrt(2 * log(p) );
    signal_est_now = mean( x_tilde_now(:, cluster_est_prev==1), 2) - mean( x_tilde_now(:, cluster_est_prev==2), 2);
    n_g1_now = sum(cluster_est_prev == 1);
    n_g2_now = sum(cluster_est_prev == 2);
    abs_diff = abs(signal_est_now)./sqrt(Omega_diag_hat) * sqrt( n_g1_now*n_g2_now/n );
    size(abs_diff);
    s_hat = abs_diff > thres; % s_hat is a p-dimensional boolean array
    
    num_selected = sum(s_hat);        % number of selected variables (true values)
    total_vars = length(s_hat);       % total number of variables
    fprintf('%d out of %d variables selected.\n', num_selected, total_vars);
end
%% select_variable_ISEE_clean
% @export
% 
% Inputs:
%% 
% * mean_now: $p\times n$ matrix of  cluster center part of the innovated data 
% matrix (pre-multiplied by precision matrix), where $p$ is the data dimension 
% and $n$ is the sample size
% * noise_now: $p\times n$ matrix of Gaussian noise part of the data matrix 
% (pre-multiplied by precision matrix), where $p$ is the data dimension and $n$ 
% is the sample size
% * Omega_diag_hat: $p$ vector of diagonal entries of precision matrix
% * cluster_est_prev: $n$ array of positive integers, where n is the sample 
% size. ex. [1 2 1 2 3 4 2 ]
%% 
% Outputs:
%% 
% * s_hat: $p$ boolean vector, where true indicates that variable is selected
function s_hat = select_variable_ISEE_clean(mean_vec, n)
    % Validate input dimensions
    [p, col_dim] = size(mean_vec);
    if col_dim ~= 2
        error('mean_vec must be a p-by-2 matrix representing class means.');
    end
    % Estimate sparse support
    mu_diff_hat = mean_vec(:,1) - mean_vec(:,2);
    threshold = 2*sqrt(log(p) * log(n) / n);
    s_hat = abs(mu_diff_hat) > threshold;  % p-dimensional boolean array
    % Print summary
    num_selected = sum(s_hat);
    sum(s_hat(1:10))
    while num_selected == 0
        threshold = threshold /2;
        s_hat = abs(mu_diff_hat) > threshold;
        num_selected = sum(s_hat);
    end
    fprintf('%d out of %d variables selected.\n', num_selected, p);
end
%% test_variable_selection_noisy
% @export
function test_variable_selection_noisy()
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
%% test_variable_selection_clean
% @export
function test_variable_selection_clean()
%TEST_ISEE_VARIABLE_SELECTION_VS_FLIP
%   Evaluates variable selection robustness to clustering error at flip ratios 0.1, 0.2, 0.3
    rng(1);
    % Parameters
    p = 800;
    n = 200;
    s = 10;
    n_trials = 20;
    flip_ratios = [0.2, 0.3, 0.4];
    [X, label_true, mu1, mu2, sep, ~, beta_star]  = generate_gaussian_data(n, p, 4, 'chain45', 1, 1/2);
    % Selection threshold
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
            get_bicluster_accuracy(cluster_estimate,label_true')
            % Run estimator
            [mean_vec, ~, ~, ~] = ISEE_bicluster_parallel(X', cluster_estimate);
            selected = select_variable_ISEE_clean(mean_vec, n);
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
%% 
% 
% 
% 
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
%% SDP clustering
% 
%% get_cov_small
% @export
% 
% Inputs:
%% 
% * x: $p\times n$ data matrix, where $p$ is the data dimension and $n$ is the 
% sample size
% * cluster_est_now: $n$ array of positive integers, where n is the sample size. 
% ex. [1 2 1 2 3 4 2 ]
% * s_hat: $p$ boolean vector, where true indicates that variable is selected
%% 
% Outputs:
%% 
% * Sigma_hat_s_hat_now: 
function Sigma_hat_s_hat_now = get_cov_small(x, cluster_est, s_hat)
    % Inputs:
    %   x           - p × n data matrix
    %   cluster_est - n × 1 vector of cluster labels (1 or 2)
    %   s_hat       - p × 1 logical vector selecting variables (features)
    % Ensure s_hat is a column vector
    s_hat = s_hat(:);  
    
    % Split by cluster
    X_g1_now = x(:, cluster_est == 1); 
    X_g2_now = x(:, cluster_est == 2); 
    % Mean center each group
    X_mean_g1_now = mean(X_g1_now, 2);
    X_mean_g2_now = mean(X_g2_now, 2);
    % Residuals (centered data from both clusters)
    data_py = [(X_g1_now - X_mean_g1_now), (X_g2_now - X_mean_g2_now)];  % p × n
    % Select variables using s_hat
    data_filtered = data_py(s_hat, :);  % s × n
    % Compute covariance matrix (transpose to n × s)
    Sigma_hat_s_hat_now = cov(data_filtered');
end
%% cluster_SDP_noniso
% @export
% 
% Performs the clustering step of the ISEE-based iterative method. Takes `s_hat` 
% as input, making it applicable to both types of variable selection (noisy and 
% clean). Specifically, this function implements line 6 of Algorithm 3.
% 
% $$max_{\mathbf{Z} }~\langle 	\hat{\tilde{\mathbf{X}}}_{\hat{S}^t, \cdot}^\top     
% \hat{\mathbf{\Sigma}}_{\hat{S}^t, \hat{S}^t}    \hat{\tilde{\mathbf{X}}}_{\hat{S}^t, 
% \cdot}, \mathbf{Z} \rangle$$
% 
% s.t.$\mathbf{Z} \succeq 0, 	\mathrm{tr}(\mathbf{Z}) = K, 	\mathbf{Z} \mathbf{1}_n 
% = \mathbf{1}_n,	\mathbf{Z} \geq 0$
% 
% 
% 
% inputs:
%% 
% * x: $p\times n$ data matrix, where $p$ is the data dimension and $n$ is the 
% sample size
% * K: positive integer. number of clusters.
% * mean_now: $p\times n$ matrix of  cluster center part of the innovated data 
% matrix (pre-multiplied by precision matrix), where $p$ is the data dimension 
% and $n$ is the sample size
% * noise_now: $p\times n$ matrix of Gaussian noise part of the data matrix 
% (pre-multiplied by precision matrix), where $p$ is the data dimension and $n$ 
% is the sample size
% * cluster_est_prev: $n$ array of positive integers, where n is the sample 
% size. Cluster estimate from the prevous step. ex. [1 2 1 2 3 4 2 ]
%% 
% outputs:
function [cluster_est_new, obj_sdp, obj_lik] = cluster_SDP_noniso(x, K, mean_now, noise_now, cluster_est_prev, s_hat)
    %estimate sigma hat s
    n = size(x,2)
    Sigma_hat_s_hat_now = get_cov_small(x, cluster_est_prev, s_hat);
    x_tilde_now = mean_now + noise_now;
    x_tilde_now_s  = x_tilde_now(s_hat,:);  
    affinity_matrix = x_tilde_now_s' * Sigma_hat_s_hat_now * x_tilde_now_s;
    Z = kmeans_sdp_pengwei( affinity_matrix/ n, K);
    % final thresholding
    [U_sdp,~,~] = svd(Z);
    U_top_k = U_sdp(:,1:K);
    [cluster_est_new,~] = kmeans(U_top_k,K);  % label
    cluster_est_new = cluster_est_new';    
    %objective function
    obj_sdp = 0;
    for c = 1:2
        sample_mask = cluster_est_new==c;
        obj_sdp= obj_sdp + sum(affinity_matrix(sample_mask, sample_mask), "all")/sum(sample_mask);
    end
    obj_lik = obj_sdp - sum(diag(affinity_matrix));
end
%% 
%% 
%% ISEE_kmeans_noisy_onestep
% @export
% 
% inputs
%% 
% * x: $p\times n$ data matrix, where $p$ is the data dimension and $n$ is the 
% sample size
% * K: positive integer. number of clusters.
% * cluster_est_prev: $n$ array of positive integers, where n is the sample 
% size. Cluster estimate from the prevous step. ex. [1 2 1 2 3 4 2 ]
% * is_parallel : boolean. true if using parallel computing in matlab.
%% 
% outputs
%% 
% * cluster_est_new: $n$ array of positive integers, where n is the sample size. 
% News cluster estimate. ex. [1 2 1 2 3 4 2 ]
function [cluster_est_new, obj_sdp, obj_lik]  = ISEE_kmeans_noisy_onestep(x, K, cluster_est_prev, is_parallel)
%estimation
    if is_parallel
        [~, noise_mat, Omega_diag_hat, mean_mat]  = ISEE_bicluster_parallel(x, cluster_est_prev);
    else
        [~, noise_mat, Omega_diag_hat, mean_mat]  = ISEE_bicluster(x, cluster_est_prev);
    end
%variable selection
    s_hat = select_variable_ISEE_noisy(mean_mat, noise_mat, Omega_diag_hat, cluster_est_prev);
%clustering
    [cluster_est_new, obj_sdp, obj_lik]  = cluster_SDP_noniso(x, K, mean_mat, noise_mat, cluster_est_prev, s_hat);
end
%% ISEE_kmeans_noisy
% @export
% 
% inputs
%% 
% * x: $p\times n$ data matrix, where $p$ is the data dimension and $n$ is the 
% sample size
% * K: positive integer. number of clusters.
% * is_parallel : boolean. true if using parallel computing in matlab.
%% 
% outputs
%% 
% * cluster_est: $n$ array of positive integers, where n is the sample size. 
% estimated cluster size. ex. [1 2 1 2 1 2 2 ]
function cluster_estimate = ISEE_kmeans_noisy(x, k, n_iter, is_parallel)
%initialization
    cluster_estimate = cluster_spectral(x, k);
    for iter = 1:n_iter
        cluster_estimate = ISEE_kmeans_noisy_onestep(x, k, cluster_estimate, is_parallel);
    end
end
%% 
%% ISEE_kmeans_clean_onestep
% @export
% 
% inputs
%% 
% * x: $p\times n$ data matrix, where $p$ is the data dimension and $n$ is the 
% sample size
% * K: positive integer. number of clusters.
% * cluster_est_prev: $n$ array of positive integers, where n is the sample 
% size. Cluster estimate from the prevous step. ex. [1 2 1 2 3 4 2 ]
% * is_parallel : boolean. true if using parallel computing in matlab.
%% 
% outputs
%% 
% * cluster_est_new: $n$ array of positive integers, where n is the sample size. 
% News cluster estimate. ex. [1 2 1 2 3 4 2 ]
function [cluster_est_new, s_hat, obj_sdp, obj_lik]  = ISEE_kmeans_clean_onestep(x, K, cluster_est_prev, is_parallel)
%estimation
    if is_parallel
        [mean_vec, noise_mat, Omega_diag_hat, mean_mat]  = ISEE_bicluster_parallel(x, cluster_est_prev);
    else
        [mean_vec, noise_mat, Omega_diag_hat, mean_mat]  = ISEE_bicluster(x, cluster_est_prev);
    end
%variable selection
    n= size(x,2);
    s_hat = select_variable_ISEE_clean(mean_vec, n);
%clustering
    [cluster_est_new, obj_sdp, obj_lik]  = cluster_SDP_noniso(x, K, mean_mat, noise_mat, cluster_est_prev, s_hat);
end
%% 
%% ISEE_kmeans_clean
% @export
% 
% 
% 
% % ISEE_kmeans_clean - Iterative clustering using ISEE-based refinement and 
% early stopping
% 
% %
% 
% % Inputs:
% 
% %   x                - Data matrix (p × n)
% 
% %   k                - Number of clusters
% 
% %   n_iter           - Maximum number of iterations
% 
% %   is_parallel      - Logical flag for parallel execution
% 
% %   loop_detect_start - Iteration to start loop detection
% 
% %   window_size      - Number of steps used for stagnation detection
% 
% %   min_delta        - Minimum improvement required to continue iterating
% 
% %
% 
% % Output:
% 
% %   cluster_estimate - Final cluster assignment (1 × n)
% 
% 
function cluster_estimate = ISEE_kmeans_clean(x, k, n_iter, is_parallel, loop_detect_start, window_size, min_delta)
    % Initialize tracking vectors
    obj_sdp = nan(1, n_iter);
    obj_lik = nan(1, n_iter);
    % Initial cluster assignment using spectral clustering
    cluster_estimate = cluster_spectral(x, k);
    for iter = 1:n_iter
        % One step of ISEE-based k-means refinement
        [cluster_estimate, s_hat,  obj_sdp(iter), obj_lik(iter)]  = ISEE_kmeans_clean_onestep(x, k, cluster_estimate, is_parallel);
        fprintf('Iteration %d | SDP obj: %.4f | Likelihood obj: %.4f\n', iter, obj_sdp(iter), obj_lik(iter));
        % Compute objective values
        
        % Early stopping condition
        is_stop = decide_stop(obj_sdp, obj_lik, loop_detect_start, window_size, min_delta);
        if is_stop
            break;
        end
    end
end
%% Algorithm - stopping criterion
% 
%% get_sdp_objective
% @export
% 
% Computes the SDP objective 
% 
% *Inputs:* 
%% 
% * X: p x n data matrix (usually a truncated matrix, where p is |S| where S 
% is selected variables) )
% * G: 1 x n vector of cluster labels in {1,...,K}
%% 
% 
% 
% 
function obj = get_sdp_objective(X, G)
A = X' * X;        % n x n Gram matrix
K = max(G);        % number of clusters
obj_sum = 0;
for k = 1:K
    idx = find(G == k);
    A_sub = A(idx, idx);
    obj_sum = obj_sum + sum(A_sub, 'all') / numel(idx);
end
obj = 2 * obj_sum;
end
%% 
% 
%% get_likelihood_objective
% @export
% 
% *Inputs:* 
%% 
% * X: p x n data matrix (usually a truncated matrix, where p is |S| where S 
% is selected variables) )
% * G: 1 x n vector of cluster labels in {1,...,K}
function obj = get_likelihood_objective(X, G)
% Computes the full profile likelihood objective
% X: p x n data matrix
% G: 1 x n vector of cluster labels
sdp_obj = get_sdp_objective(X, G);      % reuse core SDP component
frob_norm_sq = norm(X, 'fro')^2;
obj = sdp_obj - 2 * frob_norm_sq;
end
%% 
% 
%% get_penalized_objective
% @export
% 
% *Inputs:* 
%% 
% * X: p x n data matrix (usually a truncated matrix, where p is |S| where S 
% is selected variables) )
% * G: 1 x n vector of cluster labels in {1,...,K}
function obj = get_penalized_objective(X, G)
% Computes the penalized objective combining the profile likelihood 
% and squared L2 distance between cluster means.
%
% Inputs:
%   X : p x n data matrix
%   G : 1 x n vector of cluster labels
%
% Output:
%   obj : scalar value of penalized objective
    [p, n] = size(X);
    n1 = sum(G==1);
    n2 = sum(G==2);
    sd_noise_entry = (n / (n1*n2));
    % Reuse core likelihood component
    lik_obj = get_likelihood_objective(X, G);    
    % Compute cluster means
    cluster_mean_one = mean(X(:, G == 1), 2);  % p x 1
    cluster_mean_two = mean(X(:, G == 2), 2);  % p x 1
    % Compute squared L2 distance between cluster means
    diff = cluster_mean_one - cluster_mean_two;
    penalty = n * sum(diff .^ 2) / sd_noise_entry;
    % Combine likelihood and penalty
    obj = lik_obj + penalty;
end
%% compare_cluster_support_distributions
% @export
function compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, true_support, false_support, n_rep, beta_seed)
% Compare objective value distributions for likelihood and SDP
% under 20%, 40%, and 50% label flips. Plot 8 histograms (2x4 layout).
flip_rates = [0.2, 0.4, 0.5, 1];
% Generate a general random cluster estimate (1 or 2)
cluster_rand = randi([1, 2], n, 1);
[X_full, y_true, ~, ~, ~, ~, ~] = generate_gaussian_data(n, p, s, sep, 'iso', 'random_uniform', baseline, 1, cluster_1_ratio, 1);
% Allocate results structure
results_lik = struct();
results_sdp = struct();
for f = 1:length(flip_rates)
    flip_rate = flip_rates(f);
    % Allocate
    obj_lik_tt = zeros(n_rep, 1);
    obj_lik_tr = zeros(n_rep, 1);
    obj_lik_ft = zeros(n_rep, 1);
    obj_lik_fr = zeros(n_rep, 1);
    
    obj_sdp_tt = zeros(n_rep, 1);
    obj_sdp_tr = zeros(n_rep, 1);
    obj_sdp_ft = zeros(n_rep, 1);
    obj_sdp_fr = zeros(n_rep, 1);
    
    % Flip cluster labels
        y_flip = y_true;
        n_flip = round(flip_rate * n);
        flip_idx = randperm(n, n_flip);
        y_flip(flip_idx) = 3 - y_flip(flip_idx);  % flip 1 <-> 2
        if flip_rate ==1
            y_flip = cluster_rand;
        end
    for seed = 1:n_rep
        [X_full, y_true, ~, ~, ~, ~, ~] = generate_gaussian_data(n, p, s, sep, 'iso', 'random_uniform', baseline, seed, cluster_1_ratio, beta_seed);
        X_true = X_full(:, true_support);
        X_false = X_full(:, false_support);
        % Likelihood
        obj_lik_tt(seed) = get_likelihood_objective(X_true', y_true);
        obj_lik_tr(seed) = get_likelihood_objective(X_true', y_flip);
        obj_lik_ft(seed) = get_likelihood_objective(X_false', y_true);
        obj_lik_fr(seed) = get_likelihood_objective(X_false', y_flip);
        % SDP
        obj_sdp_tt(seed) = get_sdp_objective(X_true', y_true);
        obj_sdp_tr(seed) = get_sdp_objective(X_true', y_flip);
        obj_sdp_ft(seed) = get_sdp_objective(X_false', y_true);
        obj_sdp_fr(seed) = get_sdp_objective(X_false', y_flip);
     
    end
    % Store each mode separately
    results_lik(f).flip_rate = flip_rate;
    results_lik(f).vals = {obj_lik_tt, obj_lik_tr, obj_lik_ft, obj_lik_fr};
    results_sdp(f).flip_rate = flip_rate;
    results_sdp(f).vals = {obj_sdp_tt, obj_sdp_tr, obj_sdp_ft, obj_sdp_fr};
end
fig = figure('Position', [100, 100, 1800, 900]);
t = tiledlayout(fig, 2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
labels = {'True+True', 'True+Flip', 'False+True', 'False+Flip'};
hist_handles = gobjects(1, 4);  % Store handles for common legend
for i = 1:length(flip_rates)
    % --- Likelihood Subplot ---
    nexttile(t, i);
    hold on;
    vals = results_lik(i).vals;
    for j = 1:4
        h = histogram(vals{j}, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        if i == 1  % store histogram handles from the first subplot
            hist_handles(j) = h;
        end
    end
    if flip_rates(i) == 1
        title('Likelihood | Random Guess');
    else
        title(sprintf('Likelihood | %.0f%% Flip', 100 * flip_rates(i)));
    end
    xlabel('Likelihood Objective');
    ylabel('Density');
    grid on;
    set(gca, 'FontSize', 14);
    % --- SDP Subplot ---
    nexttile(t, i + 4);
    hold on;
    vals = results_sdp(i).vals;
    for j = 1:4
        histogram(vals{j}, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    end
    if flip_rates(i) == 1
        title('SDP | Random Guess');
    else
        title(sprintf('SDP | %.0f%% Flip', 100 * flip_rates(i)));
    end
    xlabel('SDP Objective');
    ylabel('Density');
    grid on;
    set(gca, 'FontSize', 14);
end
% Add external legend below all plots
lgd = legend(hist_handles, labels, 'Orientation', 'horizontal', 'FontSize', 16);
lgd.Layout.Tile = 'south';
% Remove axes toolbar interactivity
ax_list = findall(fig, 'Type', 'axes');
for ax = ax_list'
    disableDefaultInteractivity(ax);
end
% --- Verify figure validity before saving ---
if ~isvalid(fig)
    error('Figure handle is invalid before exporting.');
end
% --- Save figure ---
fname = sprintf('objective_dists_s%d_ratio%.2f_false%d_%d.png', ...
    s, cluster_1_ratio, false_support(1), false_support(end));
fname = strrep(fname, ' ', '');
fname = strrep(fname, '[', '');
fname = strrep(fname, ']', '');
exportgraphics(fig, fname, 'Resolution', 300);
fprintf('Saved figure to: %s\n', fname);
end
%% 
% 
%% compare_cluster_support_distributions_pen
% @export
function compare_cluster_support_distributions_pen(n, p, s, sep, baseline, cluster_1_ratio, true_support, false_support, n_rep, beta_seed)
% Compare objective value distributions for likelihood and SDP
% under 20%, 40%, and 50% label flips. Plot 8 histograms (2x4 layout).
flip_rates = [0.2, 0.4, 0.5, 1];
% Generate a general random cluster estimate (1 or 2)
cluster_rand = randi([1, 2], n, 1);
[X_full, y_true, ~, ~, ~, ~, ~] = generate_gaussian_data(n, p, s, sep, 'iso', 'random_uniform', baseline, 1, cluster_1_ratio, 1);
% Allocate results structure
results_lik = struct();
results_sdp = struct();
for f = 1:length(flip_rates)
    flip_rate = flip_rates(f);
    % Allocate
    obj_lik_tt = zeros(n_rep, 1);
    obj_lik_tr = zeros(n_rep, 1);
    obj_lik_ft = zeros(n_rep, 1);
    obj_lik_fr = zeros(n_rep, 1);
    
    obj_sdp_tt = zeros(n_rep, 1);
    obj_sdp_tr = zeros(n_rep, 1);
    obj_sdp_ft = zeros(n_rep, 1);
    obj_sdp_fr = zeros(n_rep, 1);
    
    % Flip cluster labels
        y_flip = y_true;
        n_flip = round(flip_rate * n);
        flip_idx = randperm(n, n_flip);
        y_flip(flip_idx) = 3 - y_flip(flip_idx);  % flip 1 <-> 2
        if flip_rate ==1
            y_flip = cluster_rand;
        end
    for seed = 1:n_rep
        [X_full, y_true, ~, ~, ~, ~, ~] = generate_gaussian_data(n, p, s, sep, 'iso', 'random_uniform', baseline, seed, cluster_1_ratio, beta_seed);
        X_true = X_full(:, true_support);
        X_false = X_full(:, false_support);
        % Likelihood
        obj_lik_tt(seed) = get_likelihood_objective(X_true', y_true);
        obj_lik_tr(seed) = get_likelihood_objective(X_true', y_flip);
        obj_lik_ft(seed) = get_likelihood_objective(X_false', y_true);
        obj_lik_fr(seed) = get_likelihood_objective(X_false', y_flip);
        % SDP
        obj_sdp_tt(seed) = get_penalized_objective(X_true', y_true);
        obj_sdp_tr(seed) = get_penalized_objective(X_true', y_flip);
        obj_sdp_ft(seed) = get_penalized_objective(X_false', y_true);
        obj_sdp_fr(seed) = get_penalized_objective(X_false', y_flip);
     
    end
    % Store each mode separately
    results_lik(f).flip_rate = flip_rate;
    results_lik(f).vals = {obj_lik_tt, obj_lik_tr, obj_lik_ft, obj_lik_fr};
    results_sdp(f).flip_rate = flip_rate;
    results_sdp(f).vals = {obj_sdp_tt, obj_sdp_tr, obj_sdp_ft, obj_sdp_fr};
end
fig = figure('Position', [100, 100, 1800, 900]);
t = tiledlayout(fig, 2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
labels = {'True+True', 'True+Flip', 'False+True', 'False+Flip'};
hist_handles = gobjects(1, 4);  % Store handles for common legend
for i = 1:length(flip_rates)
    % --- Likelihood Subplot ---
    nexttile(t, i);
    hold on;
    vals = results_lik(i).vals;
    for j = 1:4
        h = histogram(vals{j}, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        if i == 1  % store histogram handles from the first subplot
            hist_handles(j) = h;
        end
    end
    if flip_rates(i) == 1
        title('Likelihood | Random Guess');
    else
        title(sprintf('Likelihood | %.0f%% Flip', 100 * flip_rates(i)));
    end
    xlabel('Likelihood Objective');
    ylabel('Density');
    grid on;
    set(gca, 'FontSize', 14);
    % --- SDP Subplot ---
    nexttile(t, i + 4);
    hold on;
    vals = results_sdp(i).vals;
    for j = 1:4
        histogram(vals{j}, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    end
    if flip_rates(i) == 1
        title('SDP | Random Guess');
    else
        title(sprintf('Penalized | %.0f%% Flip', 100 * flip_rates(i)));
    end
    xlabel('Penalized Objective');
    ylabel('Density');
    grid on;
    set(gca, 'FontSize', 14);
end
% Add external legend below all plots
lgd = legend(hist_handles, labels, 'Orientation', 'horizontal', 'FontSize', 16);
lgd.Layout.Tile = 'south';
% Remove axes toolbar interactivity
ax_list = findall(fig, 'Type', 'axes');
for ax = ax_list'
    disableDefaultInteractivity(ax);
end
% --- Verify figure validity before saving ---
if ~isvalid(fig)
    error('Figure handle is invalid before exporting.');
end
% --- Save figure ---
fname = sprintf('objective_dists_s%d_ratio%.2f_false%d_%d.png', ...
    s, cluster_1_ratio, false_support(1), false_support(end));
fname = strrep(fname, ' ', '');
fname = strrep(fname, '[', '');
fname = strrep(fname, ']', '');
exportgraphics(fig, fname, 'Resolution', 300);
fprintf('Saved figure to: %s\n', fname);
end
%% detect_relative_change
% @export
% 
% 
% 
% Computes the relative change in the objective value between the last two iterations.
% 
% 
% 
% *Syntax:*
% 
% |is_stuck = get_relative_change(obj_val_vec)|
% 
% 
% 
% *Input:*
%% 
% * |obj_val_vec| - Numeric vector of objective values over iterations (length 
% must be >= 2).
%% 
% *Output:*
%% 
% * |relative_change| - The relative change between the last two objective values
%% 
% *Description:*
%% 
% * This function is typically used in optimization algorithms to monitor convergence. 
% It calculates the relative difference between the two most recent objective 
% values. A small relative change suggests that the algorithm is approaching convergence. 
% * If the vector contains NaNs, the computation is based on the last two values 
% before the first NaN.
% * If fewer than two valid values exist, returns Inf.
function is_stuck = detect_relative_change(obj_val_vec, detect_start, min_delta)
% detect_relative_change - Checks whether the last two valid objective values
% show insufficient relative improvement.
%
% Inputs:
%   obj_val_vec - Vector of objective values (may contain NaNs)
%   min_delta   - Minimum required relative change to count as progress
%
% Output:
%   is_stuck    - Logical flag: true if relative change < min_delta
    % Trim at first NaN, if any
    nan_idx = find(isnan(obj_val_vec), 1, 'first');
    if isempty(nan_idx)
        valid_vals = obj_val_vec;
    else
        valid_vals = obj_val_vec(1:nan_idx - 1);
    end
    % Need at least two valid values to compute relative change
    if numel(valid_vals) < max(2, detect_start)
        is_stuck = false;
        return;
    end
    % Compute relative change
    prev_val = valid_vals(end - 1);
    curr_val = valid_vals(end);
    relative_change = abs(curr_val - prev_val) / max(abs(prev_val), eps);
    % Determine if change is below threshold
    is_stuck = relative_change < min_delta;
end
%% 
% 
% *Example:*
get_relative_change([1.0, 0.8, 0.75]) - abs((0.75 - 0.8)/0.8) 
%% 
% 
%% 
%% detect_loop
% @export
% 
% Detects convergence plateau based on recent objective values.
% 
% *Inputs:*
%% 
% * |obj_val_vec|       - Vector of objective values over iterations (may contain 
% NaN)
% * |loop_detect_start| - Number of initial steps to skip before detecting loops
% * |window_size|       - Number of recent steps to consider
% * |min_delta|    - Minimum required relative improvement (in percent) to avoid 
% detection
%% 
% *Output:*
%% 
% * |is_loop|          - Logical flag: true if no significant improvement is 
% detected
%% 
% *Description:*
%% 
% * Resembles |keras.callbacks.EarlyStopping| (<https://keras.io/api/callbacks/early_stopping/ 
% https://keras.io/api/callbacks/early_stopping/>), incorporating the |min_delta| 
% parameter. In our setting, a window of iterations plays the role of epochs in 
% deep learning.
% * If the last |window_size| iterations show no improvement over the global 
% optimum so far, return the flag |is_loop|.
%% 
% 
function is_loop = detect_loop(obj_val_vec, loop_detect_start, window_size, min_delta)
    is_loop = false; % Default output
    % Trim input at first NaN, if any
    nan_idx = find(isnan(obj_val_vec), 1, 'first');
    if ~isempty(nan_idx)
        obj_val_vec = obj_val_vec(1:nan_idx - 1);
    end
    n = numel(obj_val_vec);
    % Check if there's enough history
    if n <= loop_detect_start + window_size
        return;
    end
    % Define the best value before the recent window
    global_best = max(obj_val_vec(1:end - window_size));
    % Define the best value in the recent window
    window_vec = obj_val_vec(end - window_size + 1:end);
    window_best = max(window_vec);
    % Compute relative change
    relative_change = abs(global_best - window_best) / max(abs(global_best), eps);
    % Determine if loop (stagnation) is happening
    if relative_change < min_delta
        is_loop = true;
    end
end
% 
% *Example:*
 sequence_1 = [666.41, 1426.2, 1023.6, 1379.2,   1726, 1232.1, 1789.1, 1831.4, 1898.8, 1939.1, 1565.9, 1643.4, 1491.2, 1791.3, 1657.2, 1856.9, 1569.4, 1936.6, 1647.5, 1822.3, 1656.1, 1871.6], ...
                [-6.6188, -5.4111,  -5.713, -6.4386, -5.2692, -5.9411, -6.0581,  -6.573, -5.8132,  -6.075,  -6.481, -6.3241, -6.6013, -6.3012, -6.3302,  -5.605, -6.4672,   -6.36, -6.9014, -6.1495, -6.3179, -6.2851, -6.5507]
 detect_loop(sequence_1, 6, 5, 0.05)
sequence_2 = [1 2 3 4 5 6 7 8 9 10 NaN NaN NaN];
 detect_loop(sequence_2, 3, 3, 0.05)
%% 
% 
%% decide_stop
% @export
 function is_stop = decide_stop(obj_sdp, obj_lik, loop_detect_start, window_size, min_delta)
 is_stop = false;
        % Early stopping logic
        stop_sdp = detect_relative_change(obj_sdp, loop_detect_start, min_delta);
        stop_lik = detect_relative_change(obj_lik, loop_detect_start, min_delta);
        stagnate_sdp = detect_loop(obj_sdp, loop_detect_start, window_size, min_delta);
        stagnate_lik = detect_loop(obj_lik, loop_detect_start, window_size, min_delta);
        flags = [stop_sdp, stop_lik, stagnate_sdp, stagnate_lik];
        flag_names = {'stop_sdp', 'stop_lik', 'stagnate_sdp', 'stagnate_lik'};
        if sum(flags) >= 2
            fprintf('\nStopping early. Activated conditions:\n');
            for i = 1:length(flags)
                if flags(i)
                    fprintf('  • %s\n', flag_names{i});
                end
            end
            is_stop = true;
        end
 end
%% 
%% 
%% Simulations - data generator
%% get_precision_band
% @export
% 
% 
function precision_matrix = get_precision_band(p, precision_sparsity, conditional_correlation)
% get_precision_band - Constructs a banded symmetric precision matrix with geometric decay
%                      using spdiags, assuming identity base and symmetric off-diagonal decay.
%
% Inputs:
%   p                    - Dimension of the matrix
%   precision_sparsity   - Total number of nonzero diagonals (must be even, e.g., 2, 4, 6, ...)
%   conditional_correlation - Decay factor for off-diagonals
%
% Output:
%   precision_matrix     - p × p symmetric precision matrix
    if precision_sparsity < 2
        precision_matrix = eye(p);
        return;
    end
    max_band = floor(precision_sparsity / 2);       % e.g., 2 → 1 band above/below
    offsets = -max_band:max_band;                   % Diagonal offsets
    num_diags = length(offsets);
    % Create padded diagonal matrix: size p × num_diags
    B = zeros(p, num_diags);
    for k = 1:num_diags
        offset = offsets(k);
        len = p - abs(offset);
        decay = conditional_correlation ^ abs(offset);
        B((1:len) + max(0, offset), k) = decay;
    end
    % Build matrix from diagonals
    precision_matrix = full(spdiags(B, offsets, p, p));
end
%% get_precision_ER
% @export
% 
% Generate a precision matrix representing an Erdős–Rényi Random Graph, using 
% vectorized operations (no explicit for-loops). The random seed is fixed as |rng(1)| 
% for reproducibility. The generation procedure is as follows:
%% 
% * Let $ \tilde{\Omega} = (\tilde{\omega}_{ij})$ where 
% * $\tilde{\omega}_{ij} = u_{ij} \delta_{ij},$ where
% * $\delta_{ij} \sim \text{Bernoulli}(0.05)$ is a Bernoulli random variable 
% with success probability 0.05, 
% * $u_{ij} \sim \text{Uniform}([0.5, 1] \cup [-1, -0.5])$
% * After symmetrizing $\tilde{\Omega}$, to ensure positive definiteness, define:
% * $\Omega^* = \tilde{\Omega} + \left\{ \max\left(-\phi_{\min}(\tilde{\Omega}), 
% 0\right) + 0.05 \right\} I_p$.
% * Finally, $\Omega^*$ is standardized to have unit diagonals.
%% 
% 
function Omega_star = get_precision_ER(p)
    rng(1);  % set random seed for reproducibility
    % Get upper triangle indices (excluding diagonal)
    upper_idx = triu(true(p), 1);
    % Total number of upper triangle entries
    num_entries = sum(upper_idx(:));
    % Generate Bernoulli mask: 1 with probability 0.05
    mask = rand(num_entries, 1) < 0.01;
    % Generate random values from Unif([-1, -0.5] ∪ [0.5, 1])
    signs = 2 * (rand(num_entries, 1) < 0.5) - 1;   % ±1 with equal prob
    mags  = rand(num_entries, 1) * 0.5 + 0.5;       % ∈ [0.5, 1]
    values = signs .* mags;
    % Apply mask to only keep active edges
    values(~mask) = 0;
    % Build upper triangle of matrix
    Omega_tilde = zeros(p);
    Omega_tilde(upper_idx) = values;
    % Symmetrize
    Omega_tilde = Omega_tilde + Omega_tilde';
    % Ensure positive definiteness
    min_eig = min(eig(Omega_tilde));
    delta = max(-min_eig, 0) + 0.05;
    Omega_star = Omega_tilde + delta * eye(p);
    % Standardize to have unit diagonal
    D_inv = diag(1 ./ sqrt(diag(Omega_star)));
    Omega_star = D_inv * Omega_star * D_inv;
end
%% 
% 
%% test_get_precision_ER
% @export
function test_get_precision_ER()
%TEST_GET_PRECISION_ER Tests the get_precision_ER function for correctness
    p = 100;  % Dimensionality to test
    Omega = get_precision_ER(p);
    % Check symmetry
    if ~isequal(Omega, Omega')
        error('Test failed: Omega is not symmetric.');
    else
        disp('✓ Symmetry test passed.');
    end
    % Check positive definiteness
    eigenvalues = eig(Omega);
    if all(eigenvalues > 0)
        disp('✓ Positive definiteness test passed.');
    else
        error('Test failed: Omega is not positive definite.');
    end
    % Check unit diagonal
    diag_vals = diag(Omega);
    if max(abs(diag_vals - 1)) < 1e-10
        disp('✓ Unit diagonal test passed.');
    else
        error('Test failed: Omega is not standardized to unit diagonal.');
    end
    % Check approximate sparsity level (off-diagonal non-zeros)
    is_offdiag = ~eye(p);
    num_nonzeros_offdiag = nnz(Omega .* is_offdiag);
    expected_nonzeros = round(0.01 * p^2);
    
    if abs(num_nonzeros_offdiag - expected_nonzeros) / expected_nonzeros < 0.2
        disp(['✓ Sparsity test passed: ' num2str(num_nonzeros_offdiag) ...
              ' non-zero off-diagonal entries (expected ~' num2str(expected_nonzeros) ').']);
    else
        error(['Test failed: Unexpected number of off-diagonal non-zeros (' ...
               num2str(num_nonzeros_offdiag) ' vs expected ' num2str(expected_nonzeros) ').']);
    end
    % Optional: visualize sparsity pattern
    figure;
    spy(Omega);
    title('Sparsity Pattern of Precision Matrix \Omega^*');
    xlabel('Column');
    ylabel('Row');
end
%% 
% 
%% 
% 
% 
% 
% 
% 
%% generate_gaussian_data
% @export
% 
% 
% 
% %GENERATE_GAUSSIAN_DATA Generate two-class Gaussian data with structured precision 
% matrix
% 
% % 
% 
% *Inputs:*
% 
% %   n               - total number of samples
% 
% %   p               - number of variables
% 
% %   model           - 'ER' (Erdős–Rényi) or 'AR' (autoregressive)
% 
% %   seed            - random seed
% 
% %   cluster_1_size  - proportion of samples in class 1 (e.g., 0.5)
% 
% %
% 
% *Outputs:*
% 
% %   X           - n x p data matrix
% 
% %   y           - n x 1 vector of class labels (1 or 2)
% 
% %   Omega_star  - p x p precision matrix
% 
% %   beta_star   - sparse discriminant vector
% 
% 
% 
% 
function [X, y, mu1, mu2, mahala_dist, Omega_star, beta_star] = generate_gaussian_data(n, p, s, sep, model_cov, model_energy, baseline, seed, cluster_1_ratio, beta_seed)
  
    n1 = round(n * cluster_1_ratio);
    n2 = n - n1;
    y = [ones(n1, 1); 2 * ones(n2, 1)];
    % Generate Omega_star
    switch model_cov
        case 'iso'
            Omega_star = eye(p);
            Sigma = eye(p);
        case 'ER'
            Omega_star = get_precision_ER(p);
            Sigma = inv(Omega_star);
        case 'chain45'
            Omega_star = get_precision_band(p, 2, 0.45);
            Sigma = inv(Omega_star);
        otherwise
            error('Model must be ''ER'' or ''AR''.');
    end
    % Set beta_star
    switch model_energy
        case 'equal_symmetric'
                beta_star = zeros(p, 1);
                beta_star(1:s) = 1;
                M=sep/2/ sqrt( sum( Sigma(1:s,1:s),"all") );
                beta_star = M * beta_star;
                    % Set class means
                mu1 = Omega_star \ beta_star;
                mu2 = -mu1;
    %        signal_beta_1 = zeros(10) + 0.5;
    %        signal_beta_2 = -signal_beta_1;
        case 'random_uniform'
            rng(beta_seed)
            signal_beta_1 = 10 * rand(1, s) - 5;
            signal_beta_2 = 10 * rand(1, s) - 5;
            omega_mu_1_unscaled = [signal_beta_1, repelem(baseline, 1,p-s)];
            omega_mu_2_unscaled = [signal_beta_2, repelem(baseline, 1,p-s)];
            beta_unscaled = (omega_mu_1_unscaled - omega_mu_2_unscaled);
            sep_scale = sqrt(beta_unscaled * Sigma * beta_unscaled');            M = sep /sep_scale ;
            omega_mu_1 = M*omega_mu_1_unscaled;
            omega_mu_2 = M*omega_mu_2_unscaled;
            mu1 = Omega_star \ omega_mu_1';
            mu2 = Omega_star \ omega_mu_2';
            beta_star = omega_mu_1 - omega_mu_2;
    end
    
    
    % Mahalanobis distance
    mahala_dist_sq = (mu1 - mu2)' * Omega_star * (mu1 - mu2);
    mahala_dist = sqrt(mahala_dist_sq);
    %fprintf('Mahalanobis distance between mu1 and mu2: %.4f\n', mahala_dist);
    % Generate noise once
    rng(seed);
    Z = mvnrnd(zeros(p, 1), Sigma, n);  % n x p noise
    % Create mean matrix
    mean_matrix = [repmat(mu1', n1, 1); repmat(mu2', n2, 1)];
    % Final data matrix
    X = Z + mean_matrix;
end
%% test_generate_gaussian_data
% @export
% 
% 
function test_generate_gaussian_data()
%TEST_GENERATE_GAUSSIAN_DATA Test data generation for different p values
    n = 200;
    ps = [100, 200, 500, 800];
    seed = 42;
    cluster_1_ratio = 0.5;
    model = 'ER';
    for i = 1:length(ps)
        p = ps(i);
        fprintf('\n--- Testing with p = %d ---\n', p);
        [X, y, Omega_star, beta_star] = generate_gaussian_data(n, p, model, seed, cluster_1_ratio);
        % Check dimensions
        assert(isequal(size(X), [n, p]), 'Data matrix X has incorrect dimensions.');
        assert(isequal(size(y), [n, 1]), 'Label vector y has incorrect dimensions.');
        assert(isequal(size(Omega_star), [p, p]), 'Precision matrix has incorrect dimensions.');
        assert(isequal(size(beta_star), [p, 1]), 'Discriminant vector beta_star has incorrect dimensions.');
        % Basic checks
        fprintf('Number of samples in class 1: %d\n', sum(y == 1));
        fprintf('Number of samples in class 2: %d\n', sum(y == 2));
        fprintf('Number of non-zero entries in beta_star: %d\n', nnz(beta_star));
        % Optional: check symmetry and positive definiteness
        if ~isequal(Omega_star, Omega_star')
            warning('Omega_star is not symmetric.');
        end
        if any(eig(Omega_star) <= 0)
            warning('Omega_star is not positive definite.');
        end
    end
end
%% 
%% Simulation - auxiliary
%% get_bicluster_accuracy
% @export
%% 
% * Computes accuracy for a two-cluster assignment.
% * acc = GET_CLUSTER_ACCURACY(cluster_est, cluster_true) returns the proportion 
% of correctly assigned labels, accounting for label switching.
%% 
% *Inputs*
%% 
% * cluster_est: n array of 1 and 2
% * cluster_true: n array of 1 and 2
%% 
% Outputs
%% 
% * acc: ratio of correctly clustered observations
%% 
% 
function acc = get_bicluster_accuracy(cluster_est, cluster_true)
    % Ensure both inputs are vectors
    if ~isvector(cluster_est) || ~isvector(cluster_true)
        error('Both inputs must be vectors.');
    end
    % If one is row and one is column, transpose the row vector
    if size(cluster_est, 1) == 1 && size(cluster_true, 1) > 1
        cluster_est = cluster_est';
    elseif size(cluster_true, 1) == 1 && size(cluster_est, 1) > 1
        cluster_true = cluster_true';
    end
    if length(cluster_est) ~= length(cluster_true)
        error('Input vectors must be the same length.');
    end
    % Compute accuracy under both labelings
    match1 = sum(cluster_est == cluster_true);
    match2 = sum(cluster_est == (3 - cluster_true));  % flips 1<->2
    acc = max(match1, match2) / length(cluster_true);
end
%% 
% 
%% Simulation - step-level evaluation
% 
%% ISEE_kmeans_clean_simul
% @export
function cluster_estimate = ISEE_kmeans_clean_simul(x, k, n_iter, is_parallel, loop_detect_start, window_size, min_delta, db_dir, table_name, rep, model, sep, cluster_true)
% ISEE_kmeans_clean - Runs iterative clustering with early stopping and logs results to SQLite DB
    [p, n] = size(x);  % Get dimensions
    obj_sdp = nan(1, n_iter);
    obj_lik = nan(1, n_iter);
    % Initialize cluster assignment
    cluster_estimate = cluster_spectral(x, k);
    for iter = 1:n_iter
        [cluster_estimate, s_hat, obj_sdp(iter), obj_lik(iter)] = ISEE_kmeans_clean_onestep(x, k, cluster_estimate, is_parallel);
       %%%%%%%%%%%%%%%% simul part starts
        TP = sum(s_hat(1:10));
        FP = sum(s_hat) - TP;
        FN = 10 - TP;
acc = get_bicluster_accuracy(cluster_estimate, cluster_true);  % define this if needed
fprintf('Iteration %d | SDP obj: %.4f | Likelihood obj: %.4f | TP: %d | FP: %d | FN: %d | Acc: %.4f\n', ...
    iter, obj_sdp(iter), obj_lik(iter), TP, FP, FN, acc);
    
    % === Insert into SQLite database ===
    jobdate = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');
    max_attempts = 10;
    pause_time = 5;
    attempt = 1;
    
    while attempt <= max_attempts
        try
            conn = sqlite(db_dir, 'connect');
            insert_query = sprintf([ ...
                'INSERT INTO %s (rep, iter, sep, dim, n, model, acc, obj_sdp, obj_lik, true_pos, false_pos, false_neg, jobdate) ' ...
                'VALUES (%d, %d, %d, %d, %d, "%s", %.6f, %.6f, %.6f, %d, %d, %d, "%s")'], ...
                table_name, rep, iter, sep, p, n, model, acc, obj_sdp(iter), obj_lik(iter), TP, FP, FN, char(jobdate));
            exec(conn, insert_query);
            close(conn);
            fprintf('Inserted result successfully on attempt %d.\n', attempt);
            break;
        catch ME
            if contains(ME.message, 'database is locked')
                fprintf('Database locked. Attempt %d/%d. Retrying in %d seconds...\n', ...
                        attempt, max_attempts, pause_time);
                pause(pause_time);
                attempt = attempt + 1;
            else
                rethrow(ME);
            end
        end
    end
    if attempt > max_attempts
        error('Failed to insert after %d attempts due to persistent database lock.', max_attempts);
    end
    %%%%%%%%%%%%% simul part ends
       % Early stopping condition
        is_stop = decide_stop(obj_sdp, obj_lik, loop_detect_start, window_size, min_delta);
        if is_stop
            break;
        end
    end
end
%% 
% 
%% Baseline methods
% 
% 
% 
%% CHIME
% @export
% 
% 
% 
% Parameter estimation and clustering for a *two-class* Gaussian mixture via 
% the EM. It is based on the EM algorithm, and iteratively estimates the mixing 
% ratio omega, component means mu_1, mu_2 and beta=inv(Sigma)*(mu_1 - mu_2). This 
% code is taken directly from https://github.com/drjingma/gmm and has not been 
% modified. The |CHIME| function calls two auxiliary functions, |Contingency| 
% and |RandIndex|, which are defined immediately after it.
% 
% 
% 
% 
% 
% *Inputs*
%% 
% * z: N by p data matrix
% * zt: Nt by p training tata
% * TRUE_INDEX: true labels of the test data, used for evaluating the clustering 
% performance. If unknown, set a random index, but ignore the output aRI, RI. 
% Note if TRUE_INDEX is not available, one can input a vector consisting of all 
% ones, but the output RI, aRI should be ignored.
% * omega0: initialization for \omega
% * mu0: p x 2, initialization of [\mu_1, \mu_2]
% * beta0: p x 1, initialization of beta, 
% * rho: a vector used as constant multiplier for the penalty parameter (for 
% parameter tuning)
% * lambda: a scalar, the penalty parameter for estimating sparse beta. Default 
% is 0.1
% * maxIter: maximum number of iterations, default is 50.
% * tol: tolerance level of stability of the final estimates, default is 1e-06.
%% 
% *Outputs*
%% 
% * omega, mu, beta: estimated parameters for the Gaussian mixtures
% * RI, aRI: rand index and adjusted rand index when comparing the estimated 
% class index with the true index as vectors.  (for parameter tuning)
% * optRI, optaRI: when RI and aRI are vectors, the optimal values for RI and 
% aRI are also returned.  (for parameter tuning)
% * group_member: vector of class membership (parameter tuning applied)
function [omega, mu, beta, RI, aRI, optRI, optaRI, group_member] = CHIME(z, zt, TRUE_INDEX, omega0, mu0, beta0, rho, lambda, maxIter, tol)
if (nargin < 8), lambda = 0.1;  end
if (nargin < 9), maxIter = 50;  end
if (nargin < 10), tol = 1e-06;  end
[N,p] = size(z);
Nt = size(zt,1);
nrho = length(rho);
aRI = zeros(nrho,1);
RI = zeros(nrho,1);
omega = zeros(nrho,1);
mu = zeros(p,2,nrho);
beta = zeros(p,nrho);
IDX = zeros(Nt, nrho);
for loop_rho = 1:nrho
    lam_c = lambda + rho(loop_rho) * sqrt(log(p)/N);
    
    old_omega = omega0;
    old_mu = mu0;
    old_beta = beta0; 
    
    iter = 1;
    diff = 100;
    
    done = (diff < tol) | (iter >= maxIter);
    while (~done)
        % E-step: calculate gamma
        gamma = old_omega./((1-old_omega)*exp((z - ones(N,1)*mean(old_mu, 2)')*old_beta) + old_omega);
         
        % M-step: update omega,mu
        new_omega = mean(gamma);
        tmp1 = mean(diag(1-gamma) * z)'/(1-new_omega); 
        tmp2 = mean(diag(gamma) * z)'/new_omega;
        new_mu = [tmp1, tmp2];
        
        % Update the empirical covariance matrix Vn
        x = bsxfun(@times, sqrt(1-gamma), z-(tmp1*ones(1,N))');
        y = bsxfun(@times, sqrt(gamma), z-(tmp2*ones(1,N))');
        Vn = 1/N* (x')* x + 1/N*(y')*y;
        while cond(Vn) > 1e+6
            Vn = Vn + sqrt(log(p)/N)*diag(ones(1,p)); 
        end
        
        % M-step: update beta
        delta = tmp1 - tmp2;
        beta_init = Vn \ delta;
        % The tuning parameter in clime is updated for every iteration.
        new_beta = clime(beta_init, Vn, delta, lam_c);
        
        lam_c = 0.7 * lam_c + rho(loop_rho) * sqrt(log(p)/N);
        
        % Calculate the difference between the new value and the old value
        diff = norm(new_beta - old_beta) + norm(new_mu - old_mu) + abs(new_omega - old_omega);
             
        old_omega = new_omega;
        old_mu = new_mu;
        old_beta = new_beta;        
        iter = iter + 1;
        done = (diff < tol) | (iter >= maxIter);       
    end
    % Save the estimate
    omega(loop_rho) = new_omega;
    mu(:,:,loop_rho) = new_mu;
    beta(:,loop_rho) = new_beta;
    
    % Clustering on the test data
    IDX(:,loop_rho) = ((zt - ones( Nt,1)*mean(mu(:,:,loop_rho), 2)')*beta(:,loop_rho)>=log( omega(loop_rho)/(1-omega(loop_rho) + 1e-06) ) ) + 1;
    [aRIl,RIl,~,~] = RandIndex(IDX(:,loop_rho),TRUE_INDEX);
    aRI(loop_rho) = aRIl;
    RI(loop_rho) = RIl;
end
% optimal clustering is selected as the one that maximizes aRI
target = aRI(1);
target_index = 1;
for loop_rho = 1:nrho
    if target < aRI(loop_rho)
        target = aRI(loop_rho);
        target_index = loop_rho;
    end
end
group_member = IDX(:,target_index);
optRI = RI(target_index);
optaRI = aRI(target_index);
beta = beta(:,target_index);
mu = mu(:,:,target_index);
omega = omega(target_index);
end
%% 
% 
%% CHIME_simul_example
%% 
% * z: N by p data matrix
% * zt: Nt by p training tata
% * TRUE_INDEX: true labels of the test data, used for evaluating the clustering 
% performance. If unknown, set a random index, but ignore the output aRI, RI. 
% Note if TRUE_INDEX is not available, one can input a vector consisting of all 
% ones, but the output RI, aRI should be ignored.
% * omega0: initialization for \omega
% * mu0: p x 2, initialization of [\mu_1, \mu_2]
% * beta0: p x 1, initialization of beta, 
% * rho: a vector used as constant multiplier for the penalty parameter (for 
% parameter tuning)
% * lambda: a scalar, the penalty parameter for estimating sparse beta. Default 
% is 0.1
% * maxIter: maximum number of iterations, default is 50.
% * tol: tolerance level of stability of the final estimates, default is 1e-06.
%% test_ER_CHIME
% @export
function test_ER_CHIME()
n = 200;
p = 1000;
rep = 1;
addpath(genpath('/home1/jongminm/sparse_kmeans'));
% Set database and table
table_name = 'chime';
db_dir = '/home1/jongminm/sparse_kmeans/sparse_kmeans.db';
% Model setup
model = 'ER';
cluster_1_ratio = 0.5;
[data, label_true, mu1, mu2, sep, ~, beta_star]  = generate_gaussian_data(n, p, 4, model, rep, cluster_1_ratio);
noise_std = 1/10;
true_cluster_mean = [mu1 mu2];
noisy_cluster_mean = true_cluster_mean + randn(size(true_cluster_mean)) * noise_std;
noisy_beta = beta_star + randn(size(beta_star)) * noise_std;
lambda_multiplier = 1;
dummy_label = zeros(n,1)+1;
% Run CHIME
[~, ~, ~, ~, ~, ~, ~, cluster_est_chime] = CHIME(data, data, dummy_label, cluster_1_ratio, noisy_cluster_mean, noisy_beta,  1, 0.1,100);
% Evaluate clustering accuracy
acc = get_bicluster_accuracy(cluster_est_chime, label_true)
% Current timestamp for database
jobdate = datetime('now','Format','yyyy-MM-dd HH:mm:ss');
% Retry logic for database insertion
max_attempts = 10;
attempt = 1;
pause_time = 5;
while attempt <= max_attempts
    try
        % Open DB connection
        conn = sqlite(db_dir, 'connect');
        % Insert query
        insert_query = sprintf(['INSERT INTO %s (rep, sep, p, n, model, acc, jobdate) ' ...
                                'VALUES (%d, %.4f, %d, %d, "%s", %.6f, "%s")'], ...
                                table_name, rep, sep, p, n, model, acc, char(jobdate));
        % Execute insertion
        exec(conn, insert_query);
        close(conn);
        fprintf('Inserted result successfully on attempt %d.\n', attempt);
        break;
    catch ME
        if contains(ME.message, 'database is locked')
            fprintf('Database locked. Attempt %d/%d. Retrying in %d seconds...\n', ...
                    attempt, max_attempts, pause_time);
            pause(pause_time);
            attempt = attempt + 1;
        else
            rethrow(ME);
        end
    end
end
if attempt > max_attempts
    error('Failed to insert after %d attempts due to persistent database lock.', max_attempts);
end
end
%% 
% 
%% test_ER_isee_clean
% @export
function test_ER_isee_clean()
n = 500;
p = 400;
rep = 10;
addpath(genpath('/home1/jongminm/sparse_kmeans'));
% Set database and table
table_name = 'chime';
db_dir = '/home1/jongminm/sparse_kmeans/sparse_kmeans.db';
% Model setup
model = 'chain45';
cluster_1_ratio = 0.5;
% Generate data
[data, label_true, mu1, mu2, sep, ~, beta_star]  = generate_gaussian_data(n, p, 4, model, rep, cluster_1_ratio);
% Run our method
cluster_estimte_isee = ISEE_kmeans_clean(data', 2, 30, true, 6, 5, 0.03);
% Evaluate clustering accuracy
acc = get_bicluster_accuracy(cluster_estimte_isee, label_true)
% Current timestamp for database
jobdate = datetime('now','Format','yyyy-MM-dd HH:mm:ss');
% Retry logic for database insertion
max_attempts = 10;
attempt = 1;
pause_time = 5;
while attempt <= max_attempts
    try
        % Open DB connection
        conn = sqlite(db_dir, 'connect');
        % Insert query
        insert_query = sprintf(['INSERT INTO %s (rep, sep, p, n, model, acc, jobdate) ' ...
                                'VALUES (%d, %.4f, %d, %d, "%s", %.6f, "%s")'], ...
                                table_name, rep, sep, p, n, model, acc, char(jobdate));
        % Execute insertion
        exec(conn, insert_query);
        close(conn);
        fprintf('Inserted result successfully on attempt %d.\n', attempt);
        break;
    catch ME
        if contains(ME.message, 'database is locked')
            fprintf('Database locked. Attempt %d/%d. Retrying in %d seconds...\n', ...
                    attempt, max_attempts, pause_time);
            pause(pause_time);
            attempt = attempt + 1;
        else
            rethrow(ME);
        end
    end
end
if attempt > max_attempts
    error('Failed to insert after %d attempts due to persistent database lock.', max_attempts);
end
end
%% 
% 
%% Simulation -  auxilary
% 
%% sqlite3 table schema for baseline method
CREATE TABLE isee_new(
"rep"INTEGER,
"iter"INTEGER,
"sep"REAL,
"dim"INTEGER,
"n"INTEGER,
"model"TEXT,
"acc"REAL,
"obj_sdp"REAL,
"obj_lik"REAL,
"true_pos"INTEGER,
"false_pos"INTEGER,
"false_neg"INTEGER,
"jobdate"TIMESTAMP
);
%% 
% 
%% clime
% @export
% l1dantzig_mod.m
%
% Solves
% min_x  ||x||_1  subject to  ||Ax-b||_\infty <= epsilon
%
% Recast as linear program
% min_{x,u}  sum(u)  s.t.  x - u <= 0
%                         -x - u <= 0
%            (Ax-b) - epsilon <= 0
%            -(Ax-b) - epsilon <= 0
% and use primal-dual interior point method.
%
% Usage: xp = l1dantzig_mod(x0, A, b, epsilon, pdtol, pdmaxiter, cgtol, cgmaxiter)
%
% x0 - Nx1 vector, initial point.
%
% A - Either a handle to a function that takes a N vector and returns a K 
%     vector , or a KxN matrix.  If A is a function handle, the algorithm
%     operates in "largescale" mode, solving the Newton systems via the
%     Conjugate Gradients algorithm.
%
% At - Handle to a function that takes a K vector and returns an N vector.
%      If A is a KxN matrix, At is ignored.
%
% b - Kx1 vector of observations.
%
% epsilon - scalar or Nx1 vector of correlation constraints
%
% pdtol - Tolerance for primal-dual algorithm (algorithm terminates if
%     the duality gap is less than pdtol).  
%     Default = 1e-3.
%
% pdmaxiter - Maximum number of primal-dual iterations.  
%     Default = 50.
%
% cgtol - Tolerance for Conjugate Gradients; ignored if A is a matrix.
%     Default = 1e-8.
%
% cgmaxiter - Maximum number of iterations for Conjugate Gradients; ignored
%     if A is a matrix.
%     Default = 200.
%
% Modified by Rossi Luo (xi.rossi.luo@gmail.com), Sept 2010
function xp = clime(x0, A,  b, epsilon, pdtol, pdmaxiter, cgtol, cgmaxiter)
largescale = isa(A,'function_handle');
if (nargin < 5), pdtol = 1e-3;  end
if (nargin < 6), pdmaxiter = 50;  end
if (nargin < 7), cgtol = 1e-8;  end
if (nargin < 8), cgmaxiter = 200;  end
N = length(x0);
alpha = 0.01;
beta = 0.5;
mu = 10;
At=A;
gradf0 = [zeros(N,1); ones(N,1)];
% starting point --- make sure that it is feasible
% Now modifying it a little bit for getting feasible start
if (largescale)
  if (max( abs((A(x0) - b)) - epsilon ) > 0)
    disp('Starting point infeasible; using x0 = At*inv(AAt)*y.');
    AAt = @(z) A(At(z));
    [w, cgres] = cgsolve(AAt, b, cgtol, cgmaxiter, 0);
    if (cgres > 1/2)
      disp('A*At is ill-conditioned: cannot find starting point');
      xp = x0;
      return;
    end
    x0 = At(w);
  end
else
  if (max(abs((A*x0 - b)))>  epsilon   )
    disp('Starting point infeasible: using x0 = At*inv(AAt)*y.');
%     opts.POSDEF = true; %opts.SYM = true;
    initrho = 10*epsilon;
    nrowA = size(A,1);
    initcount = 0;
    while (max(abs((A*x0 - b)))>  epsilon   )
        [x0, hcond] = linsolve(A+eye(nrowA)*initrho, b);
        if (hcond < 1e-14) 
            disp('A*At is ill-conditioned: cannot find starting point, return initial value');
            xp = x0;
            return;
        end
        initcount = initcount + 1;
        initrho = initrho/1.2;
        if (initcount > 50) 
            break; 
        end
    end
    if (hcond < 1e-14) | initcount > 50 
      disp('A*At is ill-conditioned: cannot find starting point, return initial value');
      xp = x0;
      return;
    end
%     x0 = A'*w;
  end  
end
x = x0;
u = (0.95)*abs(x0) + (0.10)*max(abs(x0));
% set up for the first iteration
if (largescale)
  Atr = (A(x) - b);
else
  Atr = (A*x - b);
end
fu1 = x - u;
fu2 = -x - u;
fe1 = Atr - epsilon;
fe2 = -Atr - epsilon;
lamu1 = -(1./fu1);
lamu2 = -(1./fu2);
lame1 = -(1./fe1);
lame2 = -(1./fe2);
if (largescale)
  AtAv = At((lame1-lame2));
else
  AtAv = A'*((lame1-lame2));
end
% sdg = surrogate duality gap
sdg = -[fu1; fu2; fe1; fe2]'*[lamu1; lamu2; lame1; lame2];
tau = mu*(4*N)/sdg;
% residuals
rdual = gradf0 + [lamu1-lamu2 + AtAv; -lamu1-lamu2];
rcent = -[lamu1.*fu1; lamu2.*fu2; lame1.*fe1; lame2.*fe2] - (1/tau);
resnorm = norm([rdual; rcent]);
% iterations
pditer = 0;
done = (sdg < pdtol) | (pditer >= pdmaxiter);
while (~done)
  % solve for step direction
  w2 = - 1 - (1/tau)*(1./fu1 + 1./fu2);
  
  sig11 = -lamu1./fu1 - lamu2./fu2;
  sig12 = lamu1./fu1 - lamu2./fu2;
  siga = -(lame1./fe1 + lame2./fe2);
  sigx = sig11 - sig12.^2./sig11;
  
  if (largescale)
    w1 = -(1/tau)*((At(1./fe2-1./fe1)) + 1./fu2 - 1./fu1 );
    w1p = w1 - (sig12./sig11).*w2;
    hpfun = @(z) At((siga.*(A(z)))) + sigx.*z;
    [dx, cgres, cgiter] = cgsolve(hpfun, w1p, cgtol, cgmaxiter, 0);
    if (cgres > 1/2)
      disp('Cannot solve system.  Returning previous iterate.  (See Section 4 of notes for more information.)');
      xp = x;
      return
    end
    AtAdx =(A(dx));
  else
    w1 = -(1/tau)*((A'*(1./fe2-1./fe1)) + 1./fu2 - 1./fu1 );
    w1p = w1 - (sig12./sig11).*w2;
    Hp = A'*(sparse(diag(siga)))*A + diag(sigx);
    %opts.POSDEF = true; 
    opts.SYM = true;
    [dx, hcond] = linsolve(Hp, w1p,opts);
    if (hcond < 1e-14)
      disp('Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)');
      xp = x;
      return
    end
    AtAdx = (A*dx);
  end
  du = w2./sig11 - (sig12./sig11).*dx;
  
  dlamu1 = -(lamu1./fu1).*(dx-du) - lamu1 - (1/tau)*1./fu1;
  dlamu2 = -(lamu2./fu2).*(-dx-du) - lamu2 - (1/tau)*1./fu2;
  dlame1 = -(lame1./fe1).*(AtAdx) - lame1 - (1/tau)*1./fe1;
  dlame2 = -(lame2./fe2).*(-AtAdx) - lame2 - (1/tau)*1./fe2;
  if (largescale)  
    AtAdv = At((dlame1-dlame2));  
  else
    AtAdv = A'*((dlame1-dlame2));  
  end
	
  
  % find minimal step size that keeps ineq functions < 0, dual vars > 0
  iu1 = find(dlamu1 < 0); iu2 = find(dlamu2 < 0); 
  ie1 = find(dlame1 < 0); ie2 = find(dlame2 < 0);
  ifu1 = find((dx-du) > 0); ifu2 = find((-dx-du) > 0); 
  ife1 = find(AtAdx > 0); ife2 = find(-AtAdx > 0); 
  smax = min(1,min([...
    -lamu1(iu1)./dlamu1(iu1); -lamu2(iu2)./dlamu2(iu2); ...
    -lame1(ie1)./dlame1(ie1); -lame2(ie2)./dlame2(ie2); ...
    -fu1(ifu1)./(dx(ifu1)-du(ifu1)); -fu2(ifu2)./(-dx(ifu2)-du(ifu2)); ...
    -fe1(ife1)./AtAdx(ife1); -fe2(ife2)./(-AtAdx(ife2)) ]));
  s = 0.99*smax;
  
  % backtracking line search
  suffdec = 0;
  backiter = 0;
  while (~suffdec)
    xp = x + s*dx;  up = u + s*du;
    Atrp = Atr + s*AtAdx;  AtAvp = AtAv + s*AtAdv;
    fu1p = fu1 + s*(dx-du);  fu2p = fu2 + s*(-dx-du);
    fe1p = fe1 + s*AtAdx;  fe2p = fe2 + s*(-AtAdx);
    lamu1p = lamu1 + s*dlamu1;  lamu2p = lamu2 + s*dlamu2;
    lame1p = lame1 + s*dlame1; lame2p = lame2 + s*dlame2;
    rdp = gradf0 + [lamu1p-lamu2p + AtAvp; -lamu1p-lamu2p];
    rcp = -[lamu1p.*fu1p; lamu2p.*fu2p; lame1p.*fe1p; lame2p.*fe2p] - (1/tau);
    suffdec = (norm([rdp; rcp]) <= (1-alpha*s)*resnorm);
    s = beta*s;
    backiter = backiter+1;
    if (backiter > 32)
      disp('Stuck backtracking, returning last iterate.  (See Section 4 of notes for more information.)')
      xp = x;
      return
    end
  end
    
  % setup for next iteration
  x = xp;  u = up;
  Atr = Atrp;  AtAv = AtAvp;
  fu1 = fu1p; fu2 = fu2p; 
  fe1 = fe1p; fe2 = fe2p;
  lamu1 = lamu1p; lamu2 = lamu2p; 
  lame1 = lame1p; lame2 = lame2p;
  
  sdg = -[fu1; fu2; fe1; fe2]'*[lamu1; lamu2; lame1; lame2];
  tau = mu*(4*N)/sdg;
  rdual = rdp;
  rcent = -[lamu1.*fu1; lamu2.*fu2; lame1.*fe1; lame2.*fe2] - (1/tau);
  resnorm = norm([rdual; rcent]);
  
  pditer = pditer+1;
  done = (sdg < pdtol) | (pditer >= pdmaxiter);
  
%   fprintf('Iteration = %d, tau = %8.3e, Primal = %8.3e, PDGap = %8.3e, Dual res = %8.3e',...
%     pditer, tau, sum(u), sdg, norm(rdual));
% disp('\n');
%   if (largescale)
%     disp(sprintf('                CG Res = %8.3e, CG Iter = %d', cgres, cgiter));
%   else
%     disp(sprintf('                  H11p condition number = %8.3e', hcond));
%   end
  
end
%% 
% 
%% Contingency
% @export
% 
% Form contigency matrix for two vectors
% 
% C=Contingency(Mem1,Mem2) returns contingency matrix for two column vectors 
% Mem1, Mem2. These define which cluster each entity has been assigned to.
% 
% See also RANDINDEX.
% 
% (C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk
% 
% This code is taken directly from https://github.com/drjingma/gmm and has not 
% been modified. 
% 
% 
function Cont=Contingency(Mem1,Mem2)
if nargin < 2 || min(size(Mem1)) > 1 || min(size(Mem2)) > 1
   error('Contingency: Requires two vector arguments')
end
Cont=zeros(max(Mem1),max(Mem2));
for i = 1:length(Mem1);
   Cont(Mem1(i),Mem2(i))=Cont(Mem1(i),Mem2(i))+1;
end
%% 
% 
%% RandIndex
% @export
% 
% Calculates Rand Indices to compare two partitions. ARI=RANDINDEX(c1,c2), where 
% c1,c2 are vectors listing the class membership, returns the "Hubert & Arabie 
% adjusted Rand index". [AR,RI,MI,HI]=RANDINDEX(c1,c2) returns the adjusted Rand 
% index, the unadjusted Rand index, "Mirkin's" index and "Hubert's" index.
% 
% See L. Hubert and P. Arabie (1985) "Comparing Partitions" Journal of Classification 
% 2:193-218
% 
% C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk
% 
% 
function [AR,RI,MI,HI]=RandIndex(c1,c2)
if nargin < 2 || min(size(c1)) > 1 || min(size(c2)) > 1
   error('RandIndex: Requires two vector arguments');
end
C=Contingency(c1,c2);	%form contingency matrix
n=sum(sum(C));
nis=sum(sum(C,2).^2);		%sum of squares of sums of rows
njs=sum(sum(C,1).^2);		%sum of squares of sums of columns
t1=nchoosek(n,2);		%total number of pairs of entities
t2=sum(sum(C.^2));	%sum over rows & columnns of nij^2
t3=.5*(nis+njs);
%Expected index (for adjustment)
nc=(n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1));
A=t1+t2-t3;		%no. agreements
D=  -t2+t3;		%no. disagreements
if t1==nc
   AR=0;			%avoid division by zero; if k=1, define Rand = 0
else
   AR=(A-nc)/(t1-nc);		%adjusted Rand - Hubert & Arabie 1985
end
RI=A/t1;			%Rand 1971		%Probability of agreement
MI=D/t1;			%Mirkin 1970	%p(disagreement)
HI=(A-D)/t1;	%Hubert 1977	%p(agree)-p(disagree)