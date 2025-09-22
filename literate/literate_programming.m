function z = dummy(x,y)
%% DUMMY 
    z = x+y;
end
%% 
% 
%% 
% 
%% Basic functions
%% sdp_sol_to_cluster
% @export
function cluster_est = sdp_sol_to_cluster(Z_opt, K)
    [U_sdp, ~, ~] = svd(Z_opt); % extract the left singular vectors
    U_top_K = U_sdp(:, 1:K); % columns are singular vectors. extract to K. thus U_top_K is n x K (n data points, K features)
    cluster_labels = kmeans(U_top_K, K, 'Replicates', 10, 'MaxIter', 500); % Added options for robustness
    % Return cluster assignments as a row vector
    cluster_est = cluster_labels';
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
%% 
%% get_cluster_by_sdp
% @export
% 
% Solver: SDPNAL+
function cluster_est = get_cluster_by_sdp(X, K)
D = X' * X;
Z_opt = kmeans_sdp_pengwei(D, K);
cluster_est = sdp_sol_to_cluster(Z_opt, K);
end
 
%% 
%% get_cluster_by_sdp_NMF
% @export
function cluster_est = get_cluster_by_sdp_NMF(X,K)
% Initialization
n = size(X,2); % Sample size
nmX = norm(X,'fro'); % Norm of matrix X
r = 2*K; % Rank of the matrix in NNMF
alpha = 1e-6; % Step size
tol = 1e-6; % Tolerance of stopping criteria
maxiter = 50000; % Maximum of iterations
% Projection operator
proj = @(V) max(V,0) ;
% Gradient of function
grad = @(U) -4*X'*(X*U) + 4*U*(U'*U); 
% Implement algorithm
U_p = abs(randn(n,r));  U = U_p/norm(U_p,'fro')*nmX; % Random intialization
for iter = 1:maxiter
    G = grad(U);
    Unew = proj(U - alpha*G);
    
    % Evaluate iterate
    rdiff = norm(Unew - U,'fro')/norm(U,'fro');
    
    % Update the variable
    U = Unew;
% Stopping criteria
if rdiff<tol
    break
end
end
% Output matrix
U_out = U;
cluster_est = sdp_sol_to_cluster(U_out, K);
end
%% 
%% get_cluster_by_sdp_SL
% @export
function cluster_est = get_cluster_by_sdp_SL(X,K) 
    n = size(X,2); % Sample size
    p = size(X,1); % dimension
    gama = 0.1;
    columns=(rand(1,n) <gama );
    q =sum(columns);  % Random select q data points
    X_hat = X(:,columns); % New matrix with dimension p*q    
    idx_hat=get_cluster_by_sdp(X_hat, K);
 
    sumsub = histcounts(idx_hat, 1:K+1);
    C_hat=zeros(p,K);
    X_hat_1=X_hat';
 
    % Get the centers
    for cc=1:K
        findindx=find(idx_hat==cc);
        newcoln=randperm(sumsub(cc),min(sumsub));
        newindx=findindx(newcoln);
        linearIndices = newindx;
        inter=mean(X_hat_1(linearIndices,:));
        C_hat(:,cc)=inter';
    end
  
    % Assign Xi to nearesr centroid of X_hat
    cluster_est = zeros(n,1);
        for j=1:n
            fmv=zeros(1,K);
            for i=1:K
                fmv(1,i)=norm(X(:,j)-C_hat(:,i)); % Every point compared with centers
            end
            [mv,mp]=min(fmv);
        cluster_est(j)=mp; % Assigned to the position of center
    end
cluster_est = cluster_est';
end 
%% get_cluster_by_sdp_SL_NMF
% @export
function cluster_est = get_cluster_by_sdp_SL_NMF(X,K) 
    n = size(X,2); % Sample size
    p = size(X,1); % dimension
    gama = sqrt(n)/n;
    columns=(rand(1,n) <gama );
    q =sum(columns);  % Random select q data points
    X_hat = X(:,columns); % New matrix with dimension p*q    
    idx_hat=get_cluster_by_sdp_NMF(X_hat, K);
 
    sumsub = histcounts(idx_hat, 1:K+1);
    C_hat=zeros(p,K);
    X_hat_1=X_hat';
 
    % Get the centers
    for cc=1:K
        findindx=find(idx_hat==cc);
        newcoln=randperm(sumsub(cc),min(sumsub));
        newindx=findindx(newcoln);
        linearIndices = newindx;
        inter=mean(X_hat_1(linearIndices,:));
        C_hat(:,cc)=inter';
    end
  
    % Assign Xi to nearesr centroid of X_hat
    cluster_est = zeros(n,1);
        for j=1:n
            fmv=zeros(1,K);
            for i=1:K
                fmv(1,i)=norm(X(:,j)-C_hat(:,i)); % Every point compared with centers
            end
            [mv,mp]=min(fmv);
        cluster_est(j)=mp; % Assigned to the position of center
    end
cluster_est = cluster_est';
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
    p = size(x,1);
    H_hat = (x' * x)/n; %n x n  affinity matrix
    [V,D] = eig(H_hat);
    [d,ind] = sort(diag(D), "descend");
        Ds = D(ind,ind);
        
    Vs = V(:,ind);
    tau_n = 1/ log(n+p);
    delta_n = tau_n^2;
        f1 = abs(sum(V(:,1)))/sqrt(n) - 1;
    if d(1)/d(2) < 1+ tau_n
        new_data = V(:,1:2);
    elseif f1 > delta_n
        new_data = V(:,1);
    else
        new_data = V(:,2);
    end
    [cluster_est,~] = kmeans(new_data,k);
    cluster_est= cluster_est';
end
%% 
% 
% 
% We begin by implementing a single step of the algorithm, which we then use 
% to construct the full iterative procedure. Each step consists of two components: 
% variable selection and SDP-based clustering. We implement these two parts sequentially 
% and combine them into a single step function.
%% Vanilla SDP K-means
%% sdp_kmeans
% @export
% Solver: SDPNAL+
function cluster_est = sdp_kmeans(X, K)
% Input validation
if nargin < 2
    error('sdp_kmeans:NotEnoughInputs', 'Two input arguments required: data matrix X and number of clusters K.');
end
if ~ismatrix(X) || ~isnumeric(X)
    error('sdp_kmeans:InvalidX', 'Input X must be a numeric matrix.');
end
if ~isscalar(K) || K <= 1 || K ~= floor(K)
    error('GET_CLUSTER_BY_BY_SDP:InvalidK', 'Number of clusters K must be an integer greater than 1.');
end
[d, n] = size(X); % d is dimension, n is number of data points
if K > n
    error('sdp_kmeans:KExceedsN', 'Number of clusters K cannot exceed the number of data points (%d).', n);
end
 X_scaled = normalize(X');
D =  X_scaled * X_scaled';
Z_opt = kmeans_sdp_pengwei(D, K);
% Check if Z_opt is valid
if isempty(Z_opt) || ~ismatrix(Z_opt) || ~isnumeric(Z_opt)
    error('sdp_kmeans:InvalidZOpt', 'The SDP solver ''kmeans_sdp_pengwei'' returned an invalid or empty solution.');
end
% Perform eigendecomposition on the SDP solution
% The SDP solution Z_opt is often a matrix (e.g., n x n) from which
% eigenvectors are extracted for spectral clustering.
% Assuming Z_opt is an n x n matrix where n is the number of data points.
if size(Z_opt, 1) ~= n || size(Z_opt, 2) ~= n
    warning('sdp_kmeans:ZOptDimensionMismatch', ...
        'Expected Z_opt to be an %d x %d matrix, but got %d x %d. Proceeding with SVD, but results might be unexpected.', ...
        n, n, size(Z_opt, 1), size(Z_opt, 2));
end
cluster_est = sdp_sol_to_cluster(Z_opt, K);
end
 
%% 
%%  
%% Iterative algorithm: known covariance
%% sdp_kmeans_iter_knowncov
% @export
classdef sdp_kmeans_iter_knowncov < handle
    properties
        X           % Data matrix (d x n)
        K           % Number of clusters
        n           % Number of data points
        p           % Data dimension
        cutoff      % Threshold for variable inclusion
        n_iter
        time
  
    end
    methods
        function obj = sdp_kmeans_iter_knowncov(X, K)
            obj.X = X;
            obj.K = K;
            obj.n = size(X, 2);
            obj.p = size(X, 1);
            obj.n_iter = NaN;
            
            
            
        end
        
        function set_cutoff(obj)
            obj.cutoff = sqrt(2 * log(obj.p) );
        end
        function cluster_est = get_cluster(obj, X, K)
            cluster_est = get_cluster_by_sdp(X, K);
        end
        function cluster_est = get_initial_cluster(obj, X, K)
            cluster_est = get_cluster_by_sdp(X, K);
        end
        function cluster_est_now = fit_predict(obj, n_iter)     
             % written 01/11/2024
             tic
             cluster_est_now = obj.get_initial_cluster(obj.X, obj.K); % initial clustering             
             obj.set_cutoff();
             toc
            % iterate
            for iter = 1:n_iter
                fprintf("\n%i th iteration\n\n", iter)
                n_g1_now = sum(cluster_est_now == 1);
                n_g2_now = obj.n-n_g1_now;
                % 1. estimate cluster means
                if max(n_g1_now, n_g2_now) == obj.n
                    fprintf("all observations are clustered into one group")
                    cluster_est_now = repelem(1, obj.n);
                    return
                end
                
                % cluster 1 mean
                x_now_g1 = obj.X(:, (cluster_est_now ==  1));
                x_bar_g1 = mean(x_now_g1, 2);  
                % cluster 2 mean
                x_now_g2 = obj.X(:, (cluster_est_now ==  2));         
                x_bar_g2 = mean(x_now_g2, 2);
                % thresholding
                abs_diff = abs(x_bar_g1 - x_bar_g2) * sqrt( n_g1_now*n_g2_now/obj.n );
                cutoff_now = obj.cutoff;
                thresholder_vec = abs_diff > cutoff_now;
                n_selected_features = sum(thresholder_vec);
                fprintf("%i entries survived \n\n",n_selected_features)
                while n_selected_features==0 & cutoff_now>1/10
                    cutoff_now = cutoff_now*0.8;
                    thresholder_vec = abs_diff > obj.cutoff;
                    n_selected_features = sum(thresholder_vec);
                    fprintf("%i entries survived \n\n",n_selected_features)
                end
                x_sub_now = obj.X(thresholder_vec,:);
                    % 3. apply SDP k-means   
                cluster_est_now = obj.get_cluster(x_sub_now, obj.K); 
            end
            obj.time = toc
        end % end of fit_predict
    end % end of methods
end
%% sdp_kmeans_iter_knowncov_ifpca
% @export
classdef sdp_kmeans_iter_knowncov_ifpca < sdp_kmeans_iter_knowncov
       methods
    
           function cluster_est = get_cluster_initial(obj, X, K)
            cluster_est = ifpca(X, K);
        end
       end
end
%% sdp_kmeans_iter_knowncov_NMF
% @export
classdef sdp_kmeans_iter_knowncov_NMF < sdp_kmeans_iter_knowncov
       methods
    
        function obj = sdp_kmeans_iter_knowncov_NMF(X, K)
            obj = obj@sdp_kmeans_iter_knowncov(X, K);
        end        
        function cluster_est = get_cluster(obj, X, K)
            cluster_est = get_cluster_by_sdp_NMF(X, K);
        end
       end
end
%% sdp_kmeans_iter_knowncov_SL_NMF
% @export
classdef sdp_kmeans_iter_knowncov_SL_NMF < sdp_kmeans_iter_knowncov
       methods
    
        function obj = sdp_kmeans_iter_knowncov_SL_NMF(X, K)
            obj = obj@sdp_kmeans_iter_knowncov(X, K);
        end      
        
        function cluster_est = get_initial_cluster(obj, X, K)
            num_components = min(200, obj.p); % You specified 200 dimensions
            % Note: The `pca` function performs centering by default.
            % To avoid this, we'll use singular value decomposition (SVD) directly.
            [U, S, V] = svd( X', 'econ');
            data_pca = obj.X' * V(:, 1:num_components);
            cluster_est = get_cluster_by_sdp_SL_NMF(data_pca', K); % Transpose back to original format (p x n)
        end
        function cluster_est = get_cluster(obj, X, K)
            cluster_est = get_cluster_by_sdp_SL_NMF(X, K);
        end
       end
end
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
%% fit_elasticNet
% @export
function [bestBeta, bestIntercept, bestAlpha, bestMSE] = fit_elasticNet(X, y)
%TUNELASSO Fit LASSO models over a grid of alpha values and select best by CV MSE
%
%   [bestBeta, bestIntercept, bestAlpha, bestMSE] = tuneLasso(X, y)
%
%   Inputs:
%       X - n x p design matrix
%       y - n x 1 response vector
%
%   Outputs:
%       bestBeta     - best coefficient vector
%       bestIntercept- intercept corresponding to best fit
    alphas = 0.1:0.1:1;   % gamma/alpha candidates
    bestMSE = Inf;
    bestAlpha = NaN;
    bestBeta = [];
    bestIntercept = NaN;
    for a = alphas
        [B, FitInfo] = lasso(X, y, ...
            'CV', 10, ...
            'Alpha', a, ...
            'Intercept', true, ...
            'Standardize', true);
        mseVal = FitInfo.MSE(FitInfo.IndexMinMSE);
        if mseVal < bestMSE
            bestMSE = mseVal;
            bestAlpha = a;
            bestBeta = B(:, FitInfo.IndexMinMSE)';
            bestIntercept = FitInfo.Intercept(FitInfo.IndexMinMSE);
        end
    end
end
%% 
% 
% 
% 
%% get_intercept_residual_lasso_adaptive
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
function [intercept, residual] = get_intercept_residual_lasso_adaptive(response, predictor)                 
  
[intercept, slope] = fit_elasticNet(predictor,response);
 
 
    % Compute residual
    residual = response - intercept - predictor * slope;
end
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
    slowly_diverging_constant = log(log(n));
    threshold = 2*sqrt(log(p) *  slowly_diverging_constant / n);
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
%% 
%% SDP clustering
% 
% 
% 
%% 
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
%% ISEE_kmeans_clean_onestep_adaptive
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
function [cluster_est_new, s_hat, obj_sdp, obj_lik]  = ISEE_kmeans_clean_onestep_adaptive(x, K, cluster_est_prev, is_parallel)
%estimation
    if is_parallel
        [mean_vec, noise_mat, Omega_diag_hat, mean_mat]  = ISEE_bicluster_parallel_adaptive(x, cluster_est_prev);
    else
        [mean_vec, noise_mat, Omega_diag_hat, mean_mat]  = ISEE_bicluster_parallel_adaptive(x, cluster_est_prev);
    end
%variable selection
    n= size(x,2);
    s_hat = select_variable_ISEE_clean(mean_vec, n);
%clustering
    [cluster_est_new, obj_sdp, obj_lik]  = cluster_SDP_noniso(x, K, mean_mat, noise_mat, cluster_est_prev, s_hat);
end
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
%% 
%% 
%% ISEE_kmeans_clean_adaptive
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
function cluster_estimate = ISEE_kmeans_clean_adaptive(x, k, n_iter, is_parallel, loop_detect_start, window_size, min_delta)
    % Initialize tracking vectors
    obj_sdp = nan(1, n_iter);
    obj_lik = nan(1, n_iter);
    % Initial cluster assignment using spectral clustering
    cluster_estimate = cluster_spectral(x, k);
    for iter = 1:n_iter
        % One step of ISEE-based k-means refinement
        [cluster_estimate, s_hat,  obj_sdp(iter), obj_lik(iter)]  = ISEE_kmeans_clean_onestep_adaptive(x, k, cluster_estimate, is_parallel);
        fprintf('Iteration %d | SDP obj: %.4f | Likelihood obj: %.4f\n', iter, obj_sdp(iter), obj_lik(iter));
        % Compute objective values
        
        % Early stopping condition
        is_stop = decide_stop(obj_sdp, obj_lik, loop_detect_start, window_size, min_delta);
        if is_stop
            break;
        end
    end
end
%% 
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
% and squared Frobenius norm of the row-centered data matrix.
%
% Inputs:
%   X : p x n data matrix
%   G : 1 x n vector of cluster labels
%
% Output:
%   obj : scalar value of penalized objective
    [p, n] = size(X);
    
    % Core likelihood component
    lik_obj = get_likelihood_objective(X, G);    
    % Compute penalty using variance
    penalty = 2 * (n - 1) * sum(var(X, 0, 2)); 
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
%% Bandit algorithm
%% sdp_kmeans_bandit
% @export
classdef sdp_kmeans_bandit < handle
    properties
        X           % Data matrix (d x n)
        K           % Number of clusters
        n           % Number of data points
        p           % Data dimension
        cutoff      % Threshold for variable inclusion
        alpha       % Alpha parameters of Beta prior
        beta        % Beta parameters of Beta prior
        pi
        acc_dict
        signal_entry_est
        n_iter
        cluster_est
        x_tilde_est       
        omega_est_time    
        sdp_solve_time    
        entries_survived  
        obj_val_prim     
        obj_val_dual      
        obj_val_original  
    end
    methods
        function obj = sdp_kmeans_bandit(X, K)
            obj.X = X;
            obj.K = K;
            obj.n = size(X, 2);
            obj.p = size(X, 1);
            C = 0.5;
            obj.cutoff = log(1 / C) / log((1 + C) / C);
            obj.n_iter = NaN;
            
            
            
        end
        
        function set_bayesian_parameters(obj)            
            obj.alpha = ones(1, obj.p);
            obj.beta = repmat(1, 1, obj.p);
            obj.pi = obj.alpha ./ (obj.alpha + obj.beta);
        end
 
        function fit_predict(obj, n_iter)
            tic; % Start timing for the entire fit_predict method
            obj.n_iter = n_iter;
            obj.set_bayesian_parameters();
            obj.initialize_cluster_est();
            fprintf("initialization done\n")
            for i = 1:n_iter
                variable_subset_now = obj.choose();
                %disp(['Iteration ', num2str(i), ' - arms pulled: ', mat2str(find(variable_subset_now)), '\n']);
                disp(['number of arms pulled: ', mat2str(sum(variable_subset_now)), '\n']);
                reward_now = obj.reward(variable_subset_now, i);
                obj.update(variable_subset_now, reward_now);
            end
            %final clustering
            final_selection = obj.signal_entry_est;
            X_sub_final = obj.X(final_selection, :);
            obj.cluster_est = obj.get_cluster(X_sub_final, obj.K);
            % ... all existing code ...
        total_fit_predict_time = toc; % End timing for the entire fit_predict method            
        fprintf('Total fit_predict time: %.4f seconds\n', total_fit_predict_time);
        end
  
        function cluster_est = get_cluster(obj, X, K) % inherit this class and change this part to try simpler clustering methods
            cluster_est = get_cluster_by_sdp(X, K);
        end
        function initialize_cluster_est(obj)
              obj.acc_dict = containers.Map(1:(obj.n_iter+1), repelem(0, obj.n_iter+1)); 
        end
        function variable_subset = choose(obj)
            theta = betarnd(obj.alpha, obj.beta);
            variable_subset = theta > obj.cutoff;
        end
        
        function reward_vec = reward(obj, variable_subset, iter)
            % Use only selected variables
            X_sub = obj.X(variable_subset, :);
            obj.cluster_est  = obj.get_cluster(X_sub, obj.K);
            % Assume K = 2
            sample_cluster_1 = X_sub(:, obj.cluster_est == 1);
            sample_cluster_2 = X_sub(:, obj.cluster_est == 2);
            %size(sample_cluster_1, 2)
            %size(sample_cluster_2, 2)
            reward_vec = zeros(1, obj.p);
            idx = find(variable_subset);
            % only calculate the p-values for selected variables
            for j = 1:length(idx)
                i = idx(j);
                pval =  permutationTest( ...
                    sample_cluster_1(j, :), ...
                    sample_cluster_2(j, :), ...
                    100 ...
                ); % 
                reward_vec(i) = pval < 0.01;
            end
            disp(['number of rewarded pulls: ', mat2str(sum(reward_vec))]);
            
     
        end % end of method reward
        function update(obj, variable_subset, reward_vec)
            obj.alpha(variable_subset) = obj.alpha(variable_subset) + reward_vec(variable_subset);
            obj.beta(variable_subset) = obj.beta(variable_subset) + (1 - reward_vec(variable_subset));
            obj.pi = obj.alpha ./ (obj.alpha + obj.beta); 
            obj.signal_entry_est = obj.pi>0.5;
        end % end of method update    
    end % end of methods
end
%% 
%% 
% 
% 
% 
%% sdp_kmeans_bandit_simul
% @export
classdef sdp_kmeans_bandit_simul  < sdp_kmeans_bandit 
    methods
        function obj = sdp_kmeans_bandit_simul(X, number_cluster)
            % Call the superclass constructor first
            % This initializes X, K, n, p, cutoff, and n_iter properties from the superclass
            
            obj = obj@sdp_kmeans_bandit(X, number_cluster);
            
        end
        function cluster_est_final = fit_predict(obj, n_iter, cluster_true)
            obj.n_iter = n_iter;
            obj.set_bayesian_parameters();
            obj.initialize_cluster_est();
            obj.initialize_saving_matrix()
            for i = 1:n_iter
                variable_subset_now = obj.choose();
                obj.entries_survived(i, :) = variable_subset_now;
                arms_pulled = mat2str(find(variable_subset_now));
                disp(['Iteration ', num2str(i), ' - arms pulled: ', arms_pulled(1: min(20, size(arms_pulled,2)))]);
                disp(['number of arms pulled: ', mat2str(sum(variable_subset_now))]);
                reward_now = obj.reward(variable_subset_now, i);
                obj.update(variable_subset_now, reward_now);
                obj.evaluate_accuracy(obj.cluster_est, cluster_true, i);
            end
            
            %final clustering
            final_selection = obj.signal_entry_est;
 
            X_sub_final = obj.X(final_selection, :);
            obj.cluster_est = obj.get_cluster(X_sub_final, obj.K);
            obj.evaluate_accuracy(obj.cluster_est, cluster_true, obj.n_iter + 1);
            cluster_est_final = obj.cluster_est;
        end
        
        function evaluate_accuracy(obj, cluster_est, cluster_true, iter)
             obj.acc_dict(iter) = get_bicluster_accuracy(cluster_est, cluster_true);
            obj.acc_dict(iter)
        end % end of method evaluate_accuracy
        function initialize_saving_matrix(obj)
             obj.omega_est_time    = zeros(obj.n_iter, 1);
            obj.sdp_solve_time    = zeros(obj.n_iter, 1);
             obj.obj_val_prim      = zeros(obj.n_iter, 1);
            obj.obj_val_dual      = zeros(obj.n_iter, 1);
            obj.obj_val_original  = zeros(obj.n_iter, 1);
        end
      
        function database_subtable = get_database_subtable(obj, rep, Delta, support)
            s = length(support);
            current_time = get_current_time();
            [true_pos_vec, false_pos_vec, false_neg_vec, ~] = obj.evaluate_discovery(support);
            %fprintf( strcat( "acc =", join(repelem("%f ", length(acc_vec))), "\n"),  acc_vec );
            
            
             
            %values(obj.acc_dict);
            %values(cluster_string_dict);
             
            n_row = int32(obj.n_iter);
            database_subtable = table(...
                repelem(rep, n_row+1)',...                      % 01 replication number
                (1:(n_row+1))',...                              % 02 step iteration number
                repelem(Delta, n_row+1)',...                    % 03 separation
                repelem(obj.p, n_row+1)',...                    % 04 data dimension
                repelem(obj.n, n_row+1)',...                      % 05 sample size
                repelem(s, n_row+1)',...                        % 06 model
                ...
                cell2mat(values(obj.acc_dict))',...             % 07 accuracy
                ...
                repelem(0, n_row+1)',...               % 8 sdp objective function value  
                repelem(0, n_row+1)',...               % 9 likelihood value
                ...
                [0; true_pos_vec],...                           % 10 true positive
                [0; false_pos_vec],...                          % 11 false positive
                [0; false_neg_vec],...                          % 12 false negative
                ...
                repelem(current_time, n_row+1)', ...            % 13 timestamp
                'VariableNames', ...
                ...  %1      2       3      4      5        6         
                ["rep", "iter", "sep", "dim", "n", "model", ...
                ...  %7        8           9                       
                 "acc", "obj_sdp", "obj_lik",  ...
                ... % 10          11            12
                 "true_pos", "false_pos",  "false_neg",...
                ...  13
                     "jobdate"]);
        end % end of get_database_subtable
 
        function [true_pos_vec, false_pos_vec, false_neg_vec , survived_indices] = evaluate_discovery(obj, support)
            true_pos_vec  = zeros(obj.n_iter, 1);
            false_pos_vec = zeros(obj.n_iter, 1);
            false_neg_vec = zeros(obj.n_iter, 1);
            survived_indices = strings(obj.n_iter, 1);
            for i = 1:obj.n_iter
                positive_vec = obj.entries_survived(i,:);
                true_pos_vec(i)  = sum(positive_vec(support));
                false_pos_vec(i) = sum(positive_vec) - true_pos_vec(i);
    
                negative_vec = ~positive_vec;
                false_neg_vec(i) = sum(negative_vec(support));
                survived_indices(i) = get_num2str_with_mark( find(positive_vec), ',');
            end
        end % end of evaluate_discovery
  
    
    end % end of method
end % end of class
%% sdp_kmeans_bandit_sl_simul
% @export
classdef sdp_kmeans_bandit_sl_simul  < sdp_kmeans_bandit_simul 
    methods
        function obj = sdp_kmeans_bandit_sl_simul(X, number_cluster)
            % Call the superclass constructor first
            % This initializes X, K, n, p, cutoff, and n_iter properties from the superclass
            obj = obj@sdp_kmeans_bandit_simul(X, number_cluster);
            
        end
        
        function cluster_est = get_cluster(obj, X, K) % inherit this class and change this part to try simpler clustering methods
            cluster_est = get_cluster_by_sdp_SL_NMF(X, K);
        end    
 
        function reward_vec = reward(obj, variable_subset, iter)
            % Use only selected variables
            X_sub = obj.X(variable_subset, :);
            n_selected_feature = size(variable_subset,2);
            obj.cluster_est  = obj.get_cluster(X_sub, obj.K);
                            n_g1_now = sum( obj.cluster_est == 1);
                n_g2_now = obj.n-n_g1_now;
            % Assume K = 2
            sample_cluster_1 = X_sub(:, obj.cluster_est == 1);
            sample_cluster_2 = X_sub(:, obj.cluster_est == 2);
                 x_bar_g1 = mean(sample_cluster_1, 2);  
                  x_bar_g2 = mean(sample_cluster_2, 2);
            % thresholding
            reward_vec = zeros(1, obj.p);
            idx = find(variable_subset);
            abs_diff = abs(x_bar_g1 - x_bar_g2) * sqrt( n_g1_now*n_g2_now/obj.n );
                cutoff_now =   sqrt(2 * log(obj.p) );
                reward_vec(idx) = abs_diff > cutoff_now;
                n_selected_features = sum(reward_vec);
                fprintf("%i entries got a reward \n\n",n_selected_features)
              
       
            
     
        end % end of method reward
     
 
    end % end of methods
end
%% 
%% sdp_kmeans_bandit_thinning_simul
% @export
classdef sdp_kmeans_bandit_thinning_simul  < sdp_kmeans_bandit_simul 
    methods
        function obj = sdp_kmeans_bandit_thinning_simul(X, number_cluster)
            % Call the superclass constructor first
            % This initializes X, K, n, p, cutoff, and n_iter properties from the superclass
            obj = obj@sdp_kmeans_bandit_simul(X, number_cluster);
            
        end
        
    
        function reward_vec = reward(obj, variable_subset, iter)
            % Use only selected variables
            num_selected_features = sum(variable_subset);
            if num_selected_features == 0
                % If no variables selected, no reward can be computed for features.
                % All rewards are 0, and we skip clustering.
                reward_vec = zeros(1, obj.p); 
                return; % Exit early
            end
            X_sub = obj.X(variable_subset, :);
            noise_new = randn(num_selected_features, obj.n);
            X_sub_cluetering = X_sub + noise_new; 
            X_sub_variable_selection = X_sub - noise_new;
            % clustering
            obj.cluster_est = obj.get_cluster(X_sub_cluetering, obj.K);
             
            % variable selection
            A_double = get_assignment_matrix(obj.n, obj.K, obj.cluster_est);
            % Calculate the sum of each feature's values within each cluster
            % Resulting matrix 'cluster_sums' will be (num_selected_features x K)
            cluster_sums = X_sub_variable_selection * A_double; 
            cluster_sums_sq = cluster_sums.^2;
            cluster_norm = sum(cluster_sums_sq, 2)'; 
            % --- 3. Define reward_vec by thresholding cluster_norm (VECTORIZED) ---
            % Calculate the threshold
            q = obj.n * (log(obj.n) + log(obj.p)) / obj.K;
            threshold = sqrt(obj.n * q) + q;
            
            % Initialize reward_vec (full length of original features, p)
            reward_vec = zeros(1, obj.p); 
            
            % Get original indices of selected variables
            idx = find(variable_subset); 
            
            % Directly assign the thresholded values using vectorized indexed assignment
            reward_vec(idx) = cluster_norm > threshold; 
            
        end % end of method reward            
     
 
    end % end of methods
end
%% 
%% sdp_kmeans_bandit_thinning_spectral_simul
% @export
classdef sdp_kmeans_bandit_thinning_spectral_simul  < sdp_kmeans_bandit_even_simul_old 
    methods
        function obj = sdp_kmeans_bandit_thinning_spectral_simul(X, number_cluster)
            % Call the superclass constructor first
            % This initializes X, K, n, p, cutoff, and n_iter properties from the superclass
            obj = obj@sdp_kmeans_bandit_even_simul_old(X, number_cluster);
            
        end
        
    
        function cluster_est = get_cluster(obj, X, K) % inherit this class and change this part to try simpler clustering methods
            cluster_est = spectralcluster(X',K);
            cluster_est = cluster_est';
        end          
     
 
    end % end of methods
end
%% sdp_kmeans_bandit_thinning_nmf_simul
% @export
classdef sdp_kmeans_bandit_thinning_nmf_simul  < sdp_kmeans_bandit_thinning_simul 
    methods
        function obj = sdp_kmeans_bandit_thinning_nmf_simul(X, number_cluster)
            % Call the superclass constructor first
            % This initializes X, K, n, p, cutoff, and n_iter properties from the superclass
            obj = obj@sdp_kmeans_bandit_thinning_simul(X, number_cluster);
            
        end
        
    
        function cluster_est = get_cluster(obj, X, K) % inherit this class and change this part to try simpler clustering methods
             cluster_est = get_cluster_by_sdp(X, K);
         end          
     
 
    end % end of methods
end
%% sdp_kmeans_bandit_even_simul_old
% @export
classdef sdp_kmeans_bandit_even_simul_old  < sdp_kmeans_bandit_simul 
    methods
        function obj = sdp_kmeans_bandit_even_simul_old(X, number_cluster)
            % Call the superclass constructor first
            % This initializes X, K, n, p, cutoff, and n_iter properties from the superclass
            obj = obj@sdp_kmeans_bandit_simul(X, number_cluster);
            
        end
        
    
        function reward_vec = reward(obj, variable_subset, iter)
            % Use only selected variables
            num_selected_features = sum(variable_subset);
            if num_selected_features == 0
                % If no variables selected, no reward can be computed for features.
                % All rewards are 0, and we skip clustering.
                reward_vec = zeros(1, obj.p); 
                return; % Exit early
            end
            X_sub = obj.X(variable_subset, :);
            % Perform clustering on X_sub (original selected data)
            obj.cluster_est_dict(iter) = obj.get_cluster(X_sub, obj.K);
            
            % --- 1. Add standard normal matrix to X_sub (after clustering) ---
            % Dimensions of X_sub are (num_selected_features x num_data_points)
            X_sub_noisy = X_sub + randn(num_selected_features, obj.n); % Use obj.n for num_data_points
            % Get cluster estimates
            cluster_labels = obj.cluster_est_dict(iter).cluster_info_vec;
            % --- 2. Compute cluster_norm for each feature (VECTORIZED) ---
            A_double = get_assignment_matrix(obj.n, obj.K, cluster_labels);
            % Calculate the sum of each feature's values within each cluster
            % Resulting matrix 'cluster_sums' will be (num_selected_features x K)
            cluster_sums = X_sub_noisy * A_double; 
            cluster_sums_sq = cluster_sums.^2;
            cluster_norm = sum(cluster_sums_sq, 2)'; 
            % --- 3. Define reward_vec by thresholding cluster_norm (VECTORIZED) ---
            % Calculate the threshold
            q = obj.n * (log(obj.n) + log(obj.p)) / obj.K;
            threshold = sqrt(obj.n * q) + q;
            
            % Initialize reward_vec (full length of original features, p)
            reward_vec = zeros(1, obj.p); 
            
            % Get original indices of selected variables
            idx = find(variable_subset); 
            
            % Directly assign the thresholded values using vectorized indexed assignment
            reward_vec(idx) = cluster_norm > threshold; 
            
        end % end of method reward            
     
 
    end % end of methods
end
%% permutationTest
% @export
% [p, observeddifference, effectsize] = permutationTest(sample1, sample2, permutations [, varargin])
%
%       Permutation test (aka randomisation test), testing for a difference
%       in means between two samples. 
%
% In:
%       sample1 - vector of measurements from one (experimental) sample
%       sample2 - vector of measurements from a second (control) sample
%       permutations - the number of permutations
%
% Optional (name-value pairs):
%       sidedness - whether to test one- or two-sided:
%           'both' - test two-sided (default)
%           'smaller' - test one-sided, alternative hypothesis is that
%                       the mean of sample1 is smaller than the mean of
%                       sample2
%           'larger' - test one-sided, alternative hypothesis is that
%                      the mean of sample1 is larger than the mean of
%                      sample2
%       exact - whether or not to run an exact test, in which all possible
%               combinations are considered. this is only feasible for
%               relatively small sample sizes. the 'permutations' argument
%               will be ignored for an exact test. (1|0, default 0)
%       plotresult - whether or not to plot the distribution of randomised
%                    differences, along with the observed difference (1|0,
%                    default: 0)
%       showprogress - whether or not to show a progress bar. if 0, no bar
%                      is displayed; if showprogress > 0, the bar updates 
%                      every showprogress-th iteration.
%
% Out:  
%       p - the resulting p-value
%       observeddifference - the observed difference between the two
%                            samples, i.e. mean(sample1) - mean(sample2)
%       effectsize - the effect size, Hedges' g
%
% Usage example:
%       >> permutationTest(rand(1,100), rand(1,100)-.25, 10000, ...
%          'plotresult', 1, 'showprogress', 250)
% 
%                    Copyright 2015-2018, 2021 Laurens R Krol
%                    Team PhyPA, Biological Psychology and Neuroergonomics,
%                    Berlin Institute of Technology
% 2021-01-13 lrk
%   - Replaced effect size calculation with Hedges' g, from Hedges & Olkin
%     (1985), Statistical Methods for Meta-Analysis (p. 78, formula 3),
%     Orlando, FL, USA: Academic Press.
% 2020-07-14 lrk
%   - Added version-dependent call to hist/histogram
% 2019-02-01 lrk
%   - Added short description
%   - Increased the number of bins in the plot
% 2018-03-15 lrk
%   - Suppressed initial MATLAB:nchoosek:LargeCoefficient warning
% 2018-03-14 lrk
%   - Added exact test
% 2018-01-31 lrk
%   - Replaced calls to mean() with nanmean()
% 2017-06-15 lrk
%   - Updated waitbar message in first iteration
% 2017-04-04 lrk
%   - Added progress bar
% 2017-01-13 lrk
%   - Switched to inputParser to parse arguments
% 2016-09-13 lrk
%   - Caught potential issue when column vectors were used
%   - Improved plot
% 2016-02-17 toz
%   - Added plot functionality
% 2015-11-26 First version
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
function [p, observeddifference, effectsize] = permutationTest(sample1, sample2, permutations, varargin)
rng('shuffle');
% parsing input
p = inputParser;
addRequired(p, 'sample1', @isnumeric);
addRequired(p, 'sample2', @isnumeric);
addRequired(p, 'permutations', @isnumeric);
addParamValue(p, 'sidedness', 'both', @(x) any(validatestring(x,{'both', 'smaller', 'larger'})));
addParamValue(p, 'exact' , 0, @isnumeric);
addParamValue(p, 'plotresult', 0, @isnumeric);
addParamValue(p, 'showprogress', 0, @isnumeric);
parse(p, sample1, sample2, permutations, varargin{:})
sample1 = p.Results.sample1;
sample2 = p.Results.sample2;
permutations = p.Results.permutations;
sidedness = p.Results.sidedness;
exact = p.Results.exact;
plotresult = p.Results.plotresult;
showprogress = p.Results.showprogress;
% enforcing row vectors
if iscolumn(sample1), sample1 = sample1'; end
if iscolumn(sample2), sample2 = sample2'; end
allobservations = [sample1, sample2];
observeddifference = nanmean(sample1) - nanmean(sample2);
pooledstd = sqrt(  ( (numel(sample1)-1)*std(sample1)^2 + (numel(sample2)-1)*std(sample2)^2 )  /  ( numel(allobservations)-2 )  );
effectsize = observeddifference / pooledstd;
w = warning('off', 'MATLAB:nchoosek:LargeCoefficient');
if ~exact && permutations > nchoosek(numel(allobservations), numel(sample1))
    warning(['the number of permutations (%d) is higher than the number of possible combinations (%d);\n' ...
             'consider running an exact test using the ''exact'' argument'], ...
             permutations, nchoosek(numel(allobservations), numel(sample1)));
end
warning(w);
if showprogress, w = waitbar(0, 'Preparing test...', 'Name', 'permutationTest'); end
if exact
    % getting all possible combinations
    allcombinations = nchoosek(1:numel(allobservations), numel(sample1));
    permutations = size(allcombinations, 1);
end
% running test
randomdifferences = zeros(1, permutations);
if showprogress, waitbar(0, w, sprintf('Permutation 1 of %d', permutations), 'Name', 'permutationTest'); end
for n = 1:permutations
    if showprogress && mod(n,showprogress) == 0, waitbar(n/permutations, w, sprintf('Permutation %d of %d', n, permutations)); end
    
    % selecting either next combination, or random permutation
    if exact, permutation = [allcombinations(n,:), setdiff(1:numel(allobservations), allcombinations(n,:))];
    else, permutation = randperm(length(allobservations)); end
    
    % dividing into two samples
    randomSample1 = allobservations(permutation(1:length(sample1)));
    randomSample2 = allobservations(permutation(length(sample1)+1:length(permutation)));
    
    % saving differences between the two samples
    randomdifferences(n) = nanmean(randomSample1) - nanmean(randomSample2);
end
if showprogress, delete(w); end
% getting probability of finding observed difference from random permutations
if strcmp(sidedness, 'both')
    p = (length(find(abs(randomdifferences) > abs(observeddifference)))+1) / (permutations+1);
elseif strcmp(sidedness, 'smaller')
    p = (length(find(randomdifferences < observeddifference))+1) / (permutations+1);
elseif strcmp(sidedness, 'larger')
    p = (length(find(randomdifferences > observeddifference))+1) / (permutations+1);
end
% plotting result
if plotresult
    figure;
    if verLessThan('matlab', '8.4')
        % MATLAB R2014a and earlier
        hist(randomdifferences, 20);
    else
        % MATLAB R2014b and later
        histogram(randomdifferences, 20);
    end
    hold on;
    xlabel('Random differences');
    ylabel('Count')
    od = plot(observeddifference, 0, '*r', 'DisplayName', sprintf('Observed difference.\nEffect size: %.2f,\np = %f', effectsize, p));
    legend(od);
end
end
%% 
%% Simulations - data generator
% 
%% data_generator_subsample
% @export
classdef data_generator_subsample < handle
    properties
        X        % Data matrix (d x n)
        y
        n           % Number of data points
     percent_cluster_1
 subsample_size_cluster_1
 subsample_size_cluster_2
  
    end
    methods
    
        function obj = data_generator_subsample(X, y)
            obj.n = size(X, 2);
            obj.y = y;
             obj.X  =X;
             obj.percent_cluster_1 = sum(y==1)/sum(y>0);
        end
 
        function [X_new,y_new] = get_data(obj, subsample_size, seed)
            rng(seed);
            idx_cluster_1 = find(obj.y == 1);
            idx_cluster_2 = find(obj.y == 2); % Assuming cluster 2 is the other cluster
            
                         obj.subsample_size_cluster_1 = floor(subsample_size * obj.percent_cluster_1);
            obj.subsample_size_cluster_2 = subsample_size - obj.subsample_size_cluster_1;
            
            %pseudocode
            perm_idx_cluster_1 = randperm(numel(idx_cluster_1));
            selected_idx_cluster_1 = idx_cluster_1(perm_idx_cluster_1(1:obj.subsample_size_cluster_1));
            
            % --- Select samples from cluster 2 ---
            perm_idx_cluster_2 = randperm(numel(idx_cluster_2));
            selected_idx_cluster_2 = idx_cluster_2(perm_idx_cluster_2(1:obj.subsample_size_cluster_2));
            final_idx = [selected_idx_cluster_1, selected_idx_cluster_2];
            
            X_new =  obj.X(:,final_idx);
            y_new = obj.y(final_idx);
                 
        end
    end % end of method
    
end% end of class
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
        case 'chain45_approx'
            Omega_star = get_precision_band(p, 2, 0.45);
            diag_Omega_star = diag(Omega_star);
            Omega_star= Omega_star + delta
            Omega_star(logical(eye(size(Omega_star)))) = diag_Omega_star;
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
%% 
%% 
%% 
%% 
%% 
%% data_generator_t
% @export
classdef data_generator_t < handle
    properties
        X           % Data matrix (d x n)
        y           % cluster label
        K           % Number of clusters
        n           % Number of data points
        n1
        n2
        sep
        seed
        s
        p           % Data dimensions
        cutoff      % Threshold for variable inclusion
        n_iter
        Sigma
        precision
        mean_matrix
        noise_matrix
 
  
    end
    methods
    
        function obj = data_generator_t(n, p, s, sep, seed, cluster_1_ratio)
            obj.n = n;
            obj.p = p;
            obj.s = s;
            obj.sep = sep;
            obj.seed = seed;
            obj.n1 = round(n * cluster_1_ratio);
            obj.n2 = n - obj.n1;
            
        end
        function label = get_cluster_label(obj)
            label = [ones(obj.n1, 1); 2 * ones(obj.n2, 1)];
            label = label';
        end
        function get_cov(obj)
            obj.Sigma = eye(obj.p);
            obj.precision = obj.Sigma;
        end
        function beta_star = get_beta(obj)
             beta_star = zeros(obj.p, 1);
             beta_star(1:obj.s) = 1;
             M= (obj.sep)/2/ sqrt( sum( obj.Sigma(1:obj.s,1:obj.s),"all") );
             beta_star = M * beta_star;
        end
        function mean_matrix = get_mean_matrix(obj)
             beta  = obj.get_beta();
                    % Set class means
             mu1 = obj.precision \ beta ;
             mu2 = -mu1;
             % Create mean matrix
             mean_matrix = [repmat(mu1', obj.n1, 1); repmat(mu2', obj.n2, 1)];
             mean_matrix= mean_matrix';
        end
        function noise_matrix = get_noise_matrix(obj, df, sd)
            % Generate noise once
            rng(obj.seed);
            noise_matrix = trnd(df,[obj.p, obj.n]);  % n x p noise
            sd_for_df = sqrt( df/(df-2) );
            noise_matrix = noise_matrix * sd/sd_for_df;
 
        end
        function [X,label] = get_data(obj, df, sd)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix();
            noise_matrix = obj.get_noise_matrix(df, sd);
            X = noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
%% data_generator_t_correlated
% @export
classdef data_generator_t_correlated < data_generator_t
    methods    
        function obj = data_generator_t_correlated(n, p, s, sep, seed, cluster_1_ratio)
            obj = obj@data_generator_t(n, p, s, sep, seed, cluster_1_ratio);
        end
        function get_cov(obj)
            obj.precision = get_precision_band(obj.p, 2, 0.45);
            obj.Sigma = inv(obj.precision);
        end
        
        function noise_matrix = get_noise_matrix(obj, df, sd)
            % Generate noise once
            rng(obj.seed);
            noise_matrix = trnd(df,[obj.p, obj.n]);  % p x n noise
            sd_for_df = sqrt( df/(df-2) );
            noise_matrix = noise_matrix * sd/sd_for_df;
            noise_matrix = sqrtm(obj.Sigma) * noise_matrix;
        end
   
    end % end of method
    
end% end of class
%% 
%% data_generator_correlated_approximately_sparse_mean
% @export
classdef data_generator_correlated_approximately_sparse_mean < data_generator_t_correlated
 
    methods
    
        function obj = data_generator_correlated_approximately_sparse_mean(n, p, s, sep, seed, cluster_1_ratio)
            obj = obj@data_generator_t_correlated(n, p, s, sep, seed, cluster_1_ratio);
            
        end
        function mean_matrix = get_mean_matrix(obj, delta)
 
             mu1_primitive = obj.get_beta();
             mu2_primitive = -mu1_primitive;
             n_delta = floor(0.1*obj.p);
             mu2_primitive( (obj.s+1): (obj.s+n_delta)) = delta;
             mu1 = obj.precision \ mu1_primitive;
             mu2 = obj.precision \ mu2_primitive;
             % Create mean matrix
             mean_matrix = [repmat(mu1', obj.n1, 1); repmat(mu2', obj.n2, 1)];
             mean_matrix= mean_matrix';
             obj.mean_matrix = mean_matrix;
        end
       function noise_matrix = get_noise_matrix(obj) %modification: t noise -> Gaussian noise
            % Generate noise once
            rng(obj.seed);
            noise_matrix = mvnrnd(zeros([obj.n, obj.p]), obj.Sigma); %$Gaussian noise
            noise_matrix = noise_matrix'; % p x n matrix
            obj.noise_matrix = noise_matrix;
 
        end        
        function [X,label] = get_data(obj, delta)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix(delta);
            noise_matrix = obj.get_noise_matrix();
            X = noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
%% 
% 
%% data_generator_correlated_different_cov
% @export
classdef data_generator_correlated_different_cov < data_generator_correlated_approximately_sparse_mean
 
    methods
    
        function obj = data_generator_correlated_different_cov(n, p, s, sep, seed, cluster_1_ratio)
            obj = obj@data_generator_correlated_approximately_sparse_mean(n, p, s, sep, seed, cluster_1_ratio);
        end
 
        function [X,label] = get_data(obj, delta)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix(0);
            noise_matrix = obj.get_noise_matrix(delta);
            noise_matrix(:, 1:obj.n1      ) = (1+delta)*noise_matrix(:,1:obj.n1);
            noise_matrix(:, (obj.n1+1) : n) = (1-delta)*noise_matrix(:, (obj.n1+1) : n);
            X = noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
%% 
%% 
%% data_generator_different_cov
% @export
classdef data_generator_different_cov < data_generator_t
 
    methods
    
        function obj = data_generator_different_cov(n, p, s, sep, seed, cluster_1_ratio)
            obj = obj@data_generator_t(n, p, s, sep, seed, cluster_1_ratio);
            
        end
        function noise_matrix = get_noise_matrix(obj, sd, delta)
            % Generate noise once
            rng(obj.seed);
            noise_matrix_1 = sd*(1+delta)*normrnd(0,1,[obj.p, obj.n1]);  % p x n1 noise
            noise_matrix_2 = sd*(1-delta)*normrnd(0,1,[obj.p, obj.n2]);  % p x n2 noise
            noise_matrix = [noise_matrix_1, noise_matrix_2];
            empirical_sd_1 = std(noise_matrix_1, 0, 'all');
            empirical_sd_2 = std(noise_matrix_2, 0, 'all');
            fprintf('--- empirical_sd =%f, %f  ---\\n', empirical_sd_1, empirical_sd_2);
            
        end
        function [X,label] = get_data(obj,  sd, delta)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix();
            noise_matrix = obj.get_noise_matrix(sd, delta);
            X = noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
%% 
%% data_generator_approximately_sparse_mean
% @export
classdef data_generator_approximately_sparse_mean < data_generator_t
 
    methods
    
        function obj = data_generator_approximately_sparse_mean(n, p, s, sep, seed, cluster_1_ratio)
            obj = obj@data_generator_t(n, p, s, sep, seed, cluster_1_ratio);
            
        end
        function noise_matrix = get_noise_matrix(obj, sd)
            % Generate noise once
            rng(obj.seed);
            noise_matrix = sd*normrnd(0,1,[obj.p, obj.n]);  % p x n1 noise
            empirical_sd = std(noise_matrix, 0, 'all');
            fprintf('--- empirical_sd =%f   ---\\n', empirical_sd);
            
        end
        function [X,label] = get_data(obj, sd, delta)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix();
            n_delta = floor(obj.p * 0.1);
            mean_matrix((obj.s+1): (obj.s+n_delta), 1:obj.n1) =  delta; %approximate sparsity for cluster mean
            noise_matrix = obj.get_noise_matrix(sd);
            X = noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
%% 
%% data_generator_approximately_sparse_precision
% @export
classdef data_generator_approximately_sparse_precision < data_generator_t
    methods
        function get_cov(obj, delta)
           omat = get_precision_band(obj.p, 2, 0.45);
           [mat, rn] = findPDMatrix(omat, delta);
           rn
           obj.precision = mat;
           obj.Sigma = inv(obj.precision);
        end
    
 
        function [X,label] = get_data(obj, delta)
            obj.get_cov(delta);
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix();
             rng(obj.seed);
            X = mvnrnd(mean_matrix', obj.Sigma);
            X=X';
        end
    end % end of methods
end
 
%% 
%% data_generator_approximately_sparse_precision2
% @export
classdef data_generator_approximately_sparse_precision2 < data_generator_approximately_sparse_precision
    methods
        function get_cov(obj, delta)
           obj.precision = get_precision_band(obj.p, 2, 0.45);
           obj.precision(obj.precision == 0) = delta;     
           obj.Sigma = inv(obj.precision);
        end
    
 
 
    end % end of methods
end
 
%% 
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
    cluster_estimate = sdp_kmeans(x, k);
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
%% ISEE_kmeans_clean_simul_adaptive
% @export
function cluster_estimate = ISEE_kmeans_clean_simul_adaptive(x, k, n_iter, is_parallel, loop_detect_start, window_size, min_delta, db_dir, table_name, rep, model, sep, cluster_true)
% ISEE_kmeans_clean - Runs iterative clustering with early stopping and logs results to SQLite DB
    [p, n] = size(x);  % Get dimensions
    obj_sdp = nan(1, n_iter);
    obj_lik = nan(1, n_iter);
    % Initialize cluster assignment
    cluster_estimate = sdp_kmeans(x, k);
    for iter = 1:n_iter
        [cluster_estimate, s_hat, obj_sdp(iter), obj_lik(iter)] = ISEE_kmeans_clean_onestep_adaptive(x, k, cluster_estimate, is_parallel);
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
%% Baseline methods
%% ifpca
% @export
function [label, stats, numselect] = ifpca(Data, K, KSvalue, pvalcut, rep, kmeansrep, per)
%The function IFPCA gives an estimation of cluster labels with IF-PCA
%method according to Jin and Wang (2014).
%
%Function: [label, stats, numselect] = ifpca(Data, K, KSvalue, pvalcut, rep, kmeansrep, per)
%
%Inputs: 
%Data: p by n data matrix where p is the number of features and n is the 
%number of samples. Each column presents the observations for a 
%sample. 
%K: number of clusters
%KSvalue: simulated Kolmogorov-Smirnov statistics if possible, used to 
%estimate p-value for each feature. If left null, corresponding statistics
%will be genearated by the algorithm
%pvalcut: the threshold to elminate the effect of features as outliers.
%The default value is log(p)/p.
%rep: the number of Kolmogorov-Smirnov statistics to be simulated in the
%algorithm, defacult 100*p.
%kmeansrep: repetitions in kmeans algorithm for the last step of IF-PCA,
%defacult to be 30. 
%per: a number with 0 < per <= 1, the percentage of Kolmogorov-Smirnov 
%statistics that will be used in the normalization step, default to be 1. 
%When the data is highly skewed, this parameter can be specified, such as
%0.5.
%
%Output: 
%label: n by 1 vector, as the estimated labels for each sample
%stats: 4 by 1 struct, including the important statistics in the algorithm
%as following:
%  stats.KS: p by 1 vector shows normalized KS value for each feature;
%  stats.HC: p by 1 vector shows the HC value for each feature;
%  stats.pval: p by 1 vector shows the p-value for each feature;
%  stats.ranking: p by 1 vector shows the ranking for each feature
%    according to ranking with p-values
%numselect: number of selected features in IF-PCA
%
%Example:
% load('lungCancer.mat');
% Data = [lungCancer_test(1:149, 1:12533); lungCancertrain(:, 1:12533)];
% Data = Data';
% [p, n] = size(Data);
% [label, stats, L] = ifpca(Data, 2, [], [], 100*p, 30);
%
%Reference: 
%Jin and Wang (2014): Important Features PCA for High Dimensional
%Clustering. 
[p, n] = size(Data);
% Error checking
if (nargin<2 || isempty(K))
  error 'Please include the number of clusters'
end
if (nargin<3||isempty(KSvalue))
    nullsimu = true; 
else 
    nullsimu = false;
end
    
if (nargin<4||isempty(pvalcut))
    pvalcut = (log(p))/p;
end
if ((nargin<5||isempty(rep)) && nullsimu)
    rep = 100*p;
end
if (nargin<6||isempty(kmeansrep))
    kmeansrep = 30;
end
if (nargin<7||isempty(per))
    per = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%  Main Function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Normalize Data
gm = mean(Data'); gsd = std(Data');
Data = (Data - repmat(gm', 1, n))./repmat(gsd', 1, n);
%Simulate KS values
if(nullsimu)
KSvalue = zeros(rep,1); kk = (0:n)'/n;
for i = 1:rep
    
	x = randn(n,1); 
	z = (x - mean(x))/std(x);
	z = z/sqrt(1 - 1/n);
	pi = normcdf(z);
	pi = sort(pi);
	KSvalue(i) = max(max(abs(kk(1:n) - pi)), max(abs(kk(2:(n+1)) - pi)));
	end
KSvalue = KSvalue*sqrt(n);
clear x z pi kk;
end
KSmean = mean(KSvalue); KSstd = std(KSvalue);
if (per < 1)
    KSvalue = sort(KSvalue, 'ascend');
    KSmean = mean(KSvalue(1:round(rep*per)));
    KSstd = std(KSvalue(1:round(rep*per)));
end
%Calculate KS value for each feature in the data set
kk = (0:n)'/n;
KS = zeros(p, 1);
for j= 1:p
    pi = normcdf(Data(j,:)/sqrt(1 - 1/n));
    pi = sort(pi);
    KS(j) = sqrt(n)*max(max(abs(kk(1:n) - pi')), max(abs(kk(2:(n+1)) - pi')));
    clear pi;
end
% Standardize KS value according to Efron's idea
if (per == 1)
    KS = (KS - mean(KS))/std(KS)*KSstd + KSmean;
else 
    KS = sort(KS, 'ascend');
    KSm = mean(KS(1:round(per*p))); KSs = std(KS(1:round(per*p)));
    KS = (KS - KSm)/KSs*KSstd + KSmean;
end
% Calculate p-value with simulated KS values
pval = zeros(p,1);
for i = 1:p
    pval(i) = mean(KSvalue > KS(i));
end
[psort, ranking] = sort(pval, 'ascend');
% Calculate HC functional at each data point
kk = (1:p)'/(1 + p);
HCsort = sqrt(p)*(kk - psort)./sqrt(kk);
HCsort  =   HCsort./sqrt(max(sqrt(n)*(kk - psort)./kk, 0) + 1 );
HC = zeros(p,1);
HC(ranking) = HCsort;
% Decide the threshold
Ind = find(psort>pvalcut, 1, 'first');
ratio = HCsort;
ratio(1:Ind-1) = -Inf; ratio(round(p/2)+1:end)=-Inf;
L = find(ratio == max(ratio), 1, 'last');
numselect = L; 
% Record the statistics for every feature
stats.KS = KS; stats.HC = HC; stats.pval = pval; stats.ranking = ranking; 
% IF-PCA
data_select = Data(pval <= psort(L), :);
G = data_select'*data_select;
[V, ~] = eigs(G, K - 1); 
label = kmeans(V, K, 'replicates', kmeansrep);
end
%% randomProjectionKMeans
% @export
function cluster_est = randomProjectionKMeans(X, k, t)
%randomProjectionKMeans Implements a randomized k-means approximation algorithm.
%   Input:
%     A: n x d matrix (n points, d features)
%     k: Number of clusters
%     epsilon: Error parameter (0 < epsilon < 1/3)
%     gamma_kmeans_algo: A function handle to a gamma-approximation k-means algorithm.
%                        This function should take (data_matrix, num_clusters) as input
%                        and return an indicator matrix (n_rows x k_clusters) where
%                        X(i,j) = 1 if point i belongs to cluster j, and 0 otherwise.
%
%   Output:
%     X_gamma_tilde: Indicator matrix determining a k-partition on the rows of A_tilde.
%                    (n x k matrix)
%
%   Example usage (assuming you have a simple k-means function named 'myKMeans'):
%   % 1. Create dummy data
%   n = 100; % Number of points
%   d = 500; % Number of features
%   A = randn(n, d); % Random data
%   k = 3;   % Number of clusters
%   epsilon = 0.1;
%
%   % 2. Define a simple k-means function (replace with your actual gamma-approx algo)
%   % This example uses MATLAB's built-in kmeans, which returns cluster indices.
%   % We need to convert it to an indicator matrix format.
%   myKMeans = @(data_matrix, num_clusters) convert_labels_to_indicator(...
%                                               kmeans(data_matrix, num_clusters, ...
%                                                      'Replicates', 5, 'MaxIter', 100), ...
%                                               size(data_matrix, 1), num_clusters);
%
%   % Helper function to convert cluster labels to indicator matrix
%   function X_indicator = convert_labels_to_indicator(labels, num_points, num_clusters)
%       X_indicator = zeros(num_points, num_clusters);
%       for i = 1:num_points
%           X_indicator(i, labels(i)) = 1;
%       end
%   end
%
%   % 3. Run the random projection k-means
%   X_gamma_tilde_result = randomProjectionKMeans(A, k, epsilon, myKMeans);
%   disp('Size of the output indicator matrix:');
%   disp(size(X_gamma_tilde_result));
    
    epsilon = 0.1;
    A = X';
    % 1. Set t = Omega(k/epsilon^2)
    % A sufficiently large constant 'c' is needed. Common theoretical bounds
    % might suggest constants around 16 for certain guarantees. Let's use 16 for illustration.
    %c = 0.1;
    %t = ceil(c * k / (epsilon^2)); % Use ceil to ensure t is an integer
    %t=10;
    fprintf('Using t = %d (sketching dimension).\n', t);
    [n, d] = size(A);
    % 2. Compute a random d x t matrix R
    % Rij = +1/sqrt(t) w.p. 1/2, -1/sqrt(t) w.p. 1/2
    R = (rand(d, t) > 0.5) * (1/sqrt(t)) * 2 - (1/sqrt(t));
 
    % 3. Compute the product A_tilde = AR
    A_tilde = A * R;
    X_tilde = A_tilde';
    % 4. Run the gamma-approximation algorithm on A_tilde
    % This step assumes 'gamma_kmeans_algo' is a function handle that
    % can take A_tilde and k as input and return the indicator matrix.
    % You need to provide your specific gamma-approximation k-means algorithm here.
    cluster_est = get_cluster_by_sdp(X_tilde, k);
    fprintf('Random Projection K-Means completed.\n');
end
%% 
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
% 
% 
%% Simulation -  auxilary
% 
%% sqlite3 table schema for baseline method
CREATE TABLE table_test(
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
%% get_database_subtable
% @export
function database_subtable = get_database_subtable(rep, Delta, support, p, n, acc, time)
            s = length(support);
          
            
 
             
            n_row = 0;
            dummy = repelem(0, n_row+1)';
            database_subtable = table(...
                repelem(rep, n_row+1)',...                      % 01 replication number
                (1:(n_row+1))',...                              % 02 step iteration number
                repelem(Delta, n_row+1)',...                    % 03 separation
                repelem(p, n_row+1)',...                    % 04 data dimension
                repelem(n, n_row+1)',...                      % 05 sample size
                repelem(s, n_row+1)',...                        % 06 model
                ...
                repelem(acc, n_row+1)',...             % 07 accuracy
                ...
                dummy,...               % 8 sdp objective function value  
                dummy,...               % 9 likelihood value
                ...
                dummy,...                           % 10 true positive
                dummy,...                          % 11 false positive
                dummy,...                          % 12 false negative
                ...
                repelem(time, n_row+1)', ...            % 13 timestamp
                'VariableNames', ...
                ...  %1      2       3      4      5        6         
                ["rep", "iter", "sep", "dim", "n", "model", ...
                ...  %7        8           9                       
                 "acc", "obj_sdp", "obj_lik",  ...
                ... % 10          11            12
                 "true_pos", "false_pos",  "false_neg",...
                ...  13
                     "jobdate"]);
        end % end of get_database_subtable
%% insertTableIntoSQLite
% @export
function insertTableIntoSQLite(db_dir, table_name, database_subtable, rep, Delta, support)
max_attempts = 10;
pause_time=2;
 
attempt = 1;
while attempt <= max_attempts
    conn = []; % Initialize conn to ensure it's cleared if connection fails
    try
        % Attempt to connect to the database
        conn = sqlite(db_dir, 'connect');
        
        % *** THE CORRECT AND RECOMMENDED FIX: Use sqlwrite ***
        sqlwrite(conn, table_name, database_subtable); 
        
        % Close connection on success
        close(conn);
        fprintf('Inserted %d rows successfully into %s on attempt %d.\n', size(database_subtable, 1), table_name, attempt);
        return; % Exit the function on successful insertion
    catch ME
        % If connection was established, try to close it before retrying/rethrowing
        if ~isempty(conn) && isvalid(conn)
            close(conn); 
        end
        % Check if the error is due to database lock or busy status
        if contains(ME.message, 'database is locked', 'IgnoreCase', true) || ...
           contains(ME.message, 'SQLITE_BUSY', 'IgnoreCase', true)
            fprintf('Database locked. Attempt %d/%d. Retrying in %.1f seconds...\n', ...
                    attempt, max_attempts, pause_time);
            pause(pause_time);
            attempt = attempt + 1;
        else
            % If it's another error, rethrow it
            rethrow(ME);
        end
    end
end
% If the loop finishes without successful insertion
error('insertTableIntoSQLite:MaxAttemptsReached', 'Failed to insert after %d attempts due to persistent database lock.', max_attempts);
end
%% 
%% 
%% 
%% insertBanditIntoSQLite
% @export
function insertBanditIntoSQLite(db_dir, table_name, obj, rep, Delta, support)
% insertTableIntoSQLite Inserts a MATLAB table into an SQLite database.
%
%   insertTableIntoSQLite(db_dir, table_name, obj, rep, Delta, support, max_attempts, pause_time)
%   generates a data table using obj.get_database_subtable and attempts to
%   insert it into the specified SQLite database table. It includes a retry
%   mechanism for database lock errors.
%
%   Inputs:
%     db_dir       - (char) Full path to the SQLite database file.
%     table_name   - (char) Name of the table within the database to insert into.
%     obj          - (object) An instance of a class (e.g., sdp_kmeans_bandit_simul)
%                    that has a method called 'get_database_subtable' and other
%                    properties needed by that method.
%     rep          - (numeric) Replication number for the simulation.
%     Delta        - (numeric) Separation parameter for the simulation.
%     support      - (array) Support vector for evaluating discovery.
%     max_attempts - (numeric) Maximum number of times to retry insertion
%                    if the database is locked.
%     pause_time   - (numeric) Time in seconds to pause between retries.
%
%   Example:
%     % Assuming 'myBanditObj', 'dbPath', 'tableName', 'repNum', 'deltaVal', 'supVec'
%     % are already defined and 'dbPath' exists.
%     % Also assume 'max_attempts' = 5 and 'pause_time' = 2
%     % insertTableIntoSQLite('my_database.db', 'simulation_results', myBanditObj, ...
%     %                       1, 0.5, [1 3 5], 5, 2);
max_attempts = 10;
pause_time=2;
% Input validation (basic checks)
if ~ischar(db_dir) || isempty(db_dir)
    error('insertTableIntoSQLite:InvalidDbDir', 'db_dir must be a non-empty character array (path to database).');
end
if ~ischar(table_name) || isempty(table_name)
    error('insertTableIntoSQLite:InvalidTableName', 'table_name must be a non-empty character array.');
end
if ~isobject(obj) || ~isprop(obj, 'n_iter') % Basic check if obj is a valid object
    error('insertTableIntoSQLite:InvalidObject', 'obj must be a valid object with required properties/methods.');
end
if ~ismethod(obj, 'get_database_subtable')
    error('insertTableIntoSQLite:MissingMethod', 'The provided object ''obj'' must have a method named ''get_database_subtable''.');
end
if ~isnumeric(rep) || ~isscalar(rep)
    error('insertTableIntoSQLite:InvalidRep', 'rep must be a numeric scalar.');
end
if ~isnumeric(Delta) || ~isscalar(Delta)
    error('insertTableIntoSQLite:InvalidDelta', 'Delta must be a numeric scalar.');
end
if ~isnumeric(support) || ~isvector(support)
    error('insertTableIntoSQLite:InvalidSupport', 'support must be a numeric vector.');
end
% Generate the table using the provided object's method
fprintf('Generating database subtable...\n');
try
    database_subtable = obj.get_database_subtable(rep, Delta, support);
catch ME
    error('insertTableIntoSQLite:TableGenerationError', 'Error generating database subtable: %s', ME.message);
end
if isempty(database_subtable) || ~istable(database_subtable)
    error('insertTableIntoSQLite:EmptyOrInvalidTable', 'The generated database_subtable is empty or not a valid MATLAB table.');
end
attempt = 1;
while attempt <= max_attempts
    conn = []; % Initialize conn to ensure it's cleared if connection fails
    try
        % Attempt to connect to the database
        conn = sqlite(db_dir, 'connect');
        
        % *** THE CORRECT AND RECOMMENDED FIX: Use sqlwrite ***
        sqlwrite(conn, table_name, database_subtable); 
        
        % Close connection on success
        close(conn);
        fprintf('Inserted %d rows successfully into %s on attempt %d.\n', size(database_subtable, 1), table_name, attempt);
        return; % Exit the function on successful insertion
    catch ME
        % If connection was established, try to close it before retrying/rethrowing
        if ~isempty(conn) && isvalid(conn)
            close(conn); 
        end
        % Check if the error is due to database lock or busy status
        if contains(ME.message, 'database is locked', 'IgnoreCase', true) || ...
           contains(ME.message, 'SQLITE_BUSY', 'IgnoreCase', true)
            fprintf('Database locked. Attempt %d/%d. Retrying in %.1f seconds...\n', ...
                    attempt, max_attempts, pause_time);
            pause(pause_time);
            attempt = attempt + 1;
        else
            % If it's another error, rethrow it
            rethrow(ME);
        end
    end
end
% If the loop finishes without successful insertion
error('insertTableIntoSQLite:MaxAttemptsReached', 'Failed to insert after %d attempts due to persistent database lock.', max_attempts);
end
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
%% Miscelleonus
% 
%% get_num2str_with_mark
% @export
function num2str_with_mark = get_num2str_with_mark(integer_vec, mark)
    num2str_with_mark = regexprep(num2str(integer_vec),'\s+',mark);
end
%% 
%% cluster_est
% @export
classdef cluster_est < handle
    properties
        sample_size
        cluster_info_vec
        number_cluster
        cluster_partition
        cluster_info_string
    end
    
    methods
        function ce = cluster_est(cluster_info_vec)
            ce.sample_size = length(cluster_info_vec);
            full_index_vec = 1:ce.sample_size;
            ce.cluster_info_vec = cluster_info_vec;
            if size(ce.cluster_info_vec,2) == 1
                ce.cluster_info_vec  = ce.cluster_info_vec';
            end
            label_cluster = unique(cluster_info_vec);
            ce.number_cluster = length(label_cluster);
            % create a struct representation (which aligns with the paper)
            ce.cluster_partition = containers.Map( ...
                1:ce.number_cluster, ...
                repelem({ce.cluster_info_vec}, ce.number_cluster) ...
                );
            for i = 1:ce.number_cluster
                partition_now = full_index_vec(cluster_info_vec==i);
                ce.cluster_partition(i) = {partition_now};
            end % end of the for loop that creates the partition dictionary
        
            % create a string representation (for the database)
            ce.cluster_info_string = get_num2str_with_mark(ce.cluster_info_vec, ",");
        end %end of the constructor
    
        function acc_vec = evaluate_accuracy(ce, cluster_true)
            permutation_all = perms(1:ce.number_cluster);
            number_permutation = size(permutation_all, 1);
            acc_permutation_vec = zeros(number_permutation, 1);
            for j = 1:number_permutation
                permutation_now = permutation_all(j,:);
                cluster_permuted_now = ce.change_label(permutation_now);
                acc_permutation_vec(j) = mean(cluster_true == cluster_permuted_now);
            end % end of the for loop over permutations
            acc_vec = max( acc_permutation_vec );
        end % end of evaluate_accuracy
        function cluster_est_permuted = change_label(ce, permutation)
            cluster_est_permuted = ce.cluster_info_vec;
            for i = 1:ce.number_cluster
                cluster_est_permuted(ce.cluster_info_vec==i) = permutation(i);
            end
        end% end of change_label
    end% end of methods
end % end of the class
%% 
%% get_assignment_matrix
% @export
function A_double = get_assignment_matrix(n, K, cluster_labels)
% GET_ASSIGNMENT_MATRIX Creates a binary assignment matrix from cluster labels.
%
%   A_double = GET_ASSIGNMENT_MATRIX(n, K, cluster_labels) creates a binary
%   assignment matrix where A(j, k) = 1 if data point j belongs to cluster k,
%   and 0 otherwise. This function assumes cluster labels are integers from 1 to K.
%
%   Inputs:
%     n              - (numeric) Total number of data points.
%     K              - (numeric) Total number of clusters.
%     cluster_labels - A vector of cluster assignments for each data point.
%                      Expected to contain integer values from 1 to K.
%
%   Outputs:
%     A_double       - A (n x K) double matrix representing the
%                      binary assignment of data points to clusters.
%
%   Example:
%     n_points = 100;
%     n_clusters = 3;
%     labels = randi(n_clusters, n_points, 1); % Example labels
%     assignment_mat = get_assignment_matrix(n_points, n_clusters, labels);
%     disp(size(assignment_mat)); % Should be [100 3]
% Input validation
if ~isscalar(n) || ~isnumeric(n) || n < 1 || n ~= floor(n)
    error('get_assignment_matrix:InvalidN', 'Input ''n'' (number of data points) must be a positive integer scalar.');
end
if ~isscalar(K) || ~isnumeric(K) || K < 1 || K ~= floor(K)
    error('get_assignment_matrix:InvalidK', 'Input ''K'' (number of clusters) must be a positive integer scalar.');
end
if ~isnumeric(cluster_labels) || ~isvector(cluster_labels) || any(cluster_labels < 1) || any(cluster_labels > K) || any(cluster_labels ~= floor(cluster_labels))
    error('get_assignment_matrix:InvalidClusterLabels', 'cluster_labels must be a numeric vector of integers from 1 to K.');
end
if length(cluster_labels) ~= n
    error('get_assignment_matrix:DimensionMismatch', 'Length of cluster_labels (%d) must match n (%d).', length(cluster_labels), n);
end
% Initialize A_double as a logical matrix for efficiency
A_double = false(n, K); % A_double will be (num_data_points x K)
% Vectorized way to populate A_double based on cluster_labels
% This works if cluster_labels contains integer IDs from 1 to K
% Ensure cluster_labels is a column vector for sub2ind
linear_indices = sub2ind(size(A_double), (1:n)', cluster_labels(:));
A_double(linear_indices) = true;
% Convert to double for matrix multiplication (as required by your reward function)
A_double = double(A_double);
end
%% get_current_time
% @export
function current_time = get_current_time()
    import java.util.TimeZone 
    nn = now;
    ds = datestr(nn);
    current_time = datetime(ds,'TimeZone',char(TimeZone.getDefault().getID()));
end