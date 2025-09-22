%% ISEE_bicluster_parallel_adaptive
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
function [mean_vec, noise_mat, Omega_diag_hat, mean_mat] = ISEE_bicluster_parallel_adaptive(x, cluster_est_now)
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
   % parfor i = 1:n_regression
   for i = 1:n_regression
       i
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
                [intercept, residual] = get_intercept_residual_lasso_adaptive(response_now, predictor_now);
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
