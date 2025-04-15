function z = dummy(x,y)
%% DUMMY 
    z = x+y;
end
%% 
% 
%% Basic functions
%% get_bicluster_acc
% @export
% 
% computes the clustering accuracy between two label vectors, accounting for 
% label permutations
% 
% Inputs:
%% 
% * cluster_est: estimated labels (vector of 1s and 2s)
% * cluster_true: ground truth labels (vector of 1s and 2s)
%% 
% Output:
%% 
% * acc: clustering accuracy (between 0 and 1)
function acc = get_bicluster_acc(cluster_est, cluster_true)
    % Ensure column vectors
    cluster_est = cluster_est(:);
    cluster_true = cluster_true(:);
    if length(cluster_est) ~= length(cluster_true)
        error('Input vectors must be the same length.');
    end
    % Case 1: no permutation
    acc1 = sum(cluster_est == cluster_true);
    % Case 2: flip labels in estimated
    cluster_est_flipped = 3 - cluster_est; % flip 1<->2
    acc2 = sum(cluster_est_flipped == cluster_true);
    % Choose maximum accuracy
    acc = max(acc1, acc2) / length(cluster_true);
end
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
    cluster_est= cluster_est'
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
function [mean_now, noise_now, Omega_diag_hat] = ISEE_bicluster(x, cluster_est_now)
    p = size(x,1);
    n = size(x,2);
    n_regression = floor(p/2);
    Omega_diag_hat_even = repelem(0,p/2);
    Omega_diag_hat_odd = repelem(0,p/2);
    Omega_diag_hat = repelem(0,p);
    mean_now_even = zeros(p/2,n);
    mean_now_odd = zeros(p/2,n);
    mean_now = zeros(p,n);
    noise_now_even =zeros(p/2,n);
    noise_now_odd = zeros(p/2,n);
    noise_now = zeros(p,n);
    for i = 1 : n_regression
        alpha_Al = zeros([2,2]);
        E_Al = zeros([2,n]);
        for cluster = 1:2 %for now, the function only works for two clusters
            g_now = (cluster_est_now == cluster);
            x_noisy_g_now = x(:,g_now);
            predictor_boolean = ((1:p) == (2*(i-1)+1)) | ((1:p) == (2*(i-1)+2));
            predictor_now = x_noisy_g_now(~predictor_boolean, :)';
            for j = 1:2
                boolean_now = (1:p) == (2*(i-1)+j);
                response_now = x_noisy_g_now(boolean_now,:)';
                model_lasso = glm_gaussian(response_now, predictor_now); 
                fit = penalized(model_lasso, @p_lasso, "standardize", true);
                AIC = goodness_of_fit('aic', fit);
                [min_aic, min_aic_idx] = min(AIC);
                beta = fit.beta(:,min_aic_idx);
                slope = beta(2:end);
                intercept = beta(1);
                E_Al(j,g_now) = response_now - intercept- predictor_now * slope;
                alpha_Al(j, cluster) = intercept;
            end
        end
        %estimation
        Omega_hat_Al = inv(E_Al*E_Al')*n;% 2 x 2
        diag_Omega_hat_Al = diag(Omega_hat_Al);
        noise_Al = Omega_hat_Al*E_Al; % 2 * n
        mean_Al = zeros([2,n]);
        for cluster = 1:2
            g_now = cluster_est_now == cluster;
            n_now = sum(g_now);
            mean_Al(:,g_now) = repmat(Omega_hat_Al*alpha_Al(:,cluster), [1,n_now]);
        end
        %Omega_diag_hat( output_index ) = diag(Omega_hat_Al);
        k = i+1;
        Omega_diag_hat_odd( i ) = diag_Omega_hat_Al(1);
        Omega_diag_hat_even( i) = diag_Omega_hat_Al(2);
        mean_now_odd( i,:) = mean_Al(1,:);
        mean_now_even( i,:) = mean_Al(2,:);
        noise_now_odd( i,:) = noise_Al(1,:);
        noise_now_even( i,:) = noise_Al(2,:);
    end
    even_idx =mod((1:p),2)==0;
    odd_idx = mod((1:p),2)==1;
    Omega_diag_hat(odd_idx) = Omega_diag_hat_odd;
    Omega_diag_hat(even_idx) = Omega_diag_hat_even;
    mean_now(odd_idx,:) = mean_now_odd;
    mean_now(even_idx,:) = mean_now_even;
    noise_now(odd_idx,:) = noise_now_odd;
    noise_now(even_idx,:) = noise_now_even;
    Omega_diag_hat = Omega_diag_hat';
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
%% test_ISEE_bicluster_parallel
% @export
function test_ISEE_bicluster_parallel()
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
function s_hat = select_variable_ISEE_clean(mean_now, noise_now, cluster_est_prev)
    x_tilde_now = mean_now + noise_now;
    p = size(mean_now,1);
    n = size(mean_now,2);
    rate = sqrt(log(p)/n);
    diverging_quantity = sqrt(log(p));
    thres = diverging_quantity*rate;
    signal_est_now = mean( x_tilde_now(:, cluster_est_prev==1), 2) - mean( x_tilde_now(:, cluster_est_prev==2), 2);
    n_g1_now = sum(cluster_est_prev == 1);
    n_g2_now = sum(cluster_est_prev == 2);
    abs_diff = abs(signal_est_now)./sqrt(Omega_diag_hat) * sqrt( n_g1_now*n_g2_now/n );
    s_hat = abs_diff > thres; % s_hat is a p-dimensional boolean array
    
    num_selected = sum(s_hat);        % number of selected variables (true values)
    total_vars = length(s_hat);       % total number of variables
    fprintf('%d out of %d variables selected.\n', num_selected, total_vars);
end
%% 
%% 
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
n= size(x,2);
    X_g1_now = x(:, (cluster_est ==  1)); 
    X_g2_now = x(:, (cluster_est ==  2)); 
    X_mean_g1_now = mean(X_g1_now, 2);
    X_mean_g2_now = mean(X_g2_now, 2);
    data_py = [(X_g1_now - X_mean_g1_now) (X_g2_now - X_mean_g2_now)]; % p x n
    data_filtered = data_py(s_hat,:); % p x n
    Sigma_hat_s_hat_now = cov(data_filtered');% input: n x p 
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
function cluster_est_new = cluster_SDP_noniso(x, K, mean_now, noise_now, cluster_est_prev, s_hat)
    %estimate sigma hat s
    n = size(x,2);
    Sigma_hat_s_hat_now = get_cov_small(x, cluster_est_prev, s_hat);
    x_tilde_now = mean_now + noise_now;
    x_tilde_now_s  = x_tilde_now(s_hat,:);  
    Z_now = kmeans_sdp( x_tilde_now_s' * Sigma_hat_s_hat_now * x_tilde_now_s/ n, K);
    % final thresholding
    [U_sdp,~,~] = svd(Z_now);
    U_top_k = U_sdp(:,1:K);
    [cluster_est_new,C] = kmeans(U_top_k,K);  % label
    cluster_est_new = cluster_est_new';    
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
function cluster_est_new = ISEE_kmeans_noisy_onestep(x, K, cluster_est_prev, is_parallel)
%estimation
    if is_parallel
        [_, noise_mat, Omega_diag_hat, mean_mat]  = ISEE_bicluster_parallel(x, cluster_est_prev);
    else
        [_, noise_mat, Omega_diag_hat, mean_mat]  = ISEE_bicluster(x, cluster_est_prev);
    end
%variable selection
    s_hat = select_variable_ISEE_noisy(mean_mat, noise_mat, Omega_diag_hat, cluster_est_prev);
%clustering
    cluster_est_new = cluster_SDP_noniso(x, K, mean_mat, noise_mat, cluster_est_prev, s_hat);
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
function cluster_est = ISEE_kmeans_noisy(x, k, n_iter, is_parallel)
%initialization
    cluster_est = cluster_spectral(x, k);
    for iter = 1:n_iter
        cluster_est = ISEE_kmeans_noisy_onestep(x, k, cluster_est, is_parallel);
    end
end
%% Simulations - data generator
% 
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
    mask = rand(num_entries, 1) < 0.05;
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
    expected_nonzeros = round(0.05 * p^2);
    
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
% 
% 
% 
% 
% 
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
% % INPUTS:
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
% % OUTPUTS:
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
function [X, y, Omega_star, beta_star] = generate_gaussian_data(n, p, model, seed, cluster_1_ratio)
    rng(seed);
    s = 10;  % number of nonzero entries in beta
    n1 = round(n * cluster_1_ratio);
    n2 = n - n1;
    y = [ones(n1, 1); 2 * ones(n2, 1)];
    % Generate Omega_star
    switch model
        case 'ER'
            Omega_star = get_precision_ER(p);
        otherwise
            error('Model must be ''ER'' or ''AR''.');
    end
    % Set beta_star
    beta_star = zeros(p, 1);
    beta_star(1:s) = 1;
    % Set class means
    mu1 = zeros(p, 1);
    mu2 = mu1 - Omega_star \ beta_star;
    % Mahalanobis distance
    mahala_dist_sq = (mu1 - mu2)' * Omega_star * (mu1 - mu2);
    mahala_dist = sqrt(mahala_dist_sq);
    fprintf('Mahalanobis distance between mu1 and mu2: %.4f\n', mahala_dist);
    % Generate noise once
    Sigma = inv(Omega_star);
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