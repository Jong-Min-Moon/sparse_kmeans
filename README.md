# sparse-sdp-kmeans

To run the simulation, generate Gaussain data using the functions in the section Simulations - data generator, run the algorithm using the functions in the section SDP clustering , and evaluate the performance using the function get_bicluster_accuracy.
To run the main function, you need to install `penalized` package, `SDPNAL++` package, MATLAB statistics and machine learning toolbox. To run the ISEE subroutine in parallel, you need MATLAB parallel computing toolbox.

# Table of Contents

- Basic functions  
  - `kmeans_sdp_pengwei`

- Implementing our iterative algorithm  
  - Initialization  
    - `cluster_spectral`
  - Variable selection  
    - `ISEE_bicluster`  
    - `get_intercept_residual_lasso`  
    - `test_get_intercept_residual_lasso`  
    - `ISEE_bicluster_parallel`  
    - `select_variable_ISEE_noisy`  
    - `select_variable_ISEE_clean`  
    - `test_variable_selection_noisy`  
    - `test_variable_selection_clean`  
    - `test_variable_selection_clean_spectral`
  - SDP clustering  
    - `get_cov_small`  
    - `cluster_SDP_noniso`  
    - `ISEE_kmeans_noisy_onestep`  
    - `ISEE_kmeans_noisy`  
    - `ISEE_kmeans_clean_onestep`  
    - `ISEE_kmeans_clean`

- Algorithm - stopping criterion  
  - `get_sdp_objective`  
  - `get_likelihood_objective`  
  - `get_penalized_objective`  
  - `compare_cluster_support_distributions`  
  - `compare_cluster_support_distributions_pen`  
  - `detect_relative_change`  
  - `detect_loop`  
  - `decide_stop`

- Simulations - data generator  
  - `get_precision_band`  
  - `generate_gaussian_data`  

- Simulation - auxiliary  
  - `get_bicluster_accuracy`

- Simulation - step-level evaluation  
  - `ISEE_kmeans_clean_simul`






# Basic functions

## kmeans_sdp_pengwei
The following implementation, originally written by Mixon, Villar, and Ward and last edited on January 20, 2024, uses SDPNAL+ [3] to solve the Peng and Wei k-means SDP formulation [2], following the approach described in [1]. The original version accepts a  data matrix as input. To accommodate both isotropic and non-isotropic cases, we modify the code to accept an affinity matrix instead. Given an affinity matrix  , The code solves the following problem:
.
- **Inputs:**
  - `A`: An \(n \times n\) affinity matrix where \(n\) denotes the number of observations.
  - `k`: The number of clusters.

- **Outputs:**
  - `X`: An \(N \times N\) matrix corresponding to the solution of Peng and Wei's SDP formulation.

- **References:**
  - Mixon, Villar, Ward. *Clustering subgaussian mixtures via semidefinite programming*
  - Peng, Wei. *Approximating k-means-type clustering via semidefinite programming*
  - Yang, Sun, Toh. *SDPNAL+: A majorized semismooth Newton-CG augmented Lagrangian method for semidefinite programming with nonnegative constraints*

```matlab
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
```

# Implementing our iterative algorithm
Our iterative algorithm has the following structure:

- **Initialization**
- **For loop**
  - One iteration:
    - Variable selection  
    - SDP clustering
- **Stopping rule**

We implement these step by step.
## Initialization
For now, we only implement the spectral clustering.

### cluster_spectral

- **Inputs:**
  - `x`: Data matrix of shape \( p \times n \), where \(p\) is the data dimension and \(n\) is the sample size.
  - `k`: A positive integer specifying the number of clusters.

- **Outputs:**
  - `cluster_est`: A length-\(n\) array of positive integers indicating the cluster assignments (e.g., `[1 2 1 2 3 4 2]`).
```matlab
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
```
We begin by implementing a single step of the algorithm, which we then use to construct the full iterative procedure. Each step consists of two components: variable selection and SDP-based clustering. We implement these two parts sequentially and combine them into a single step function.
## Variable selection


### ISEE_bicluster
ISEE performs numerous Lasso regressions. We provide two versions: a plain version (this function) suitable for running on a local machine, and a parallel version (ISEE_bicluster_parallel) optimized for faster execution on computing clusters.

- **Inputs:**
  - `x`: Data matrix of shape \( p \times n \), where \(p\) is the data dimension and \(n\) is the sample size.
  - `cluster_est_now`: An array of positive integers of length \(n\), representing current cluster assignments (e.g., `[1 2 1 2 3 4 2]`).

- **Outputs:**
  - `mean_now`: A \( p \times n \) matrix representing the cluster center component of the innovated data matrix (pre-multiplied by the precision matrix).
  - `noise_now`: A \( p \times n \) matrix representing the Gaussian noise component (also pre-multiplied by the precision matrix).
  - `Omega_diag_hat`: A \( p \)-dimensional vector of the estimated diagonal entries of the precision matrix.

```matlab
function [mean_vec, noise_mat, Omega_diag_hat, mean_mat] = ISEE_bicluster(x, cluster_est_now)

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
```

### get_intercept_residual_lasso
Computes the intercept and residuals from a Lasso-penalized linear regression. Given a response vector and a predictor matrix, the predictor matrix is automatically standardized before fitting. This function fits a Lasso with many values of , selects the model with the lowest AIC, extracts the intercept and slope coefficients, and returns the residuals.

- **Inputs:**
  - `response`: An \( n \times 1 \) vector of response values.
  - `predictor`: An \( n \times p \) matrix of predictor variables.

- **Outputs:**
  - `Intercept`: A scalar intercept term from the selected Lasso model.
  - `residual`: An \( n \times 1 \) vector of residuals from the fitted model.

```matlab
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
```

### ISEE_bicluster_parallel
- **Inputs:**
  - `x`: Data matrix of shape \( p \times n \), where \(p\) is the data dimension and \(n\) is the sample size.
  - `cluster_est_now`: An array of positive integers of length \(n\), representing the current cluster assignments (e.g., `[1 2 1 2 3 4 2]`).

- **Outputs:**
  - `mean_now`: A \( p \times n \) matrix representing the cluster center component of the innovated data matrix (pre-multiplied by the precision matrix).
  - `noise_now`: A \( p \times n \) matrix representing the Gaussian noise component of the data matrix (pre-multiplied by the precision matrix).
  - `Omega_diag_hat`: A \( p \)-dimensional vec

```matlab
function [mean_vec, noise_mat, Omega_diag_hat, mean_mat] = ISEE_bicluster_parallel(x, cluster_est_now)


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
```



### select_variable_ISEE_noisy
For Algorithm 6.
- **Inputs:**
  - `mean_now`: A \( p \times n \) matrix representing the cluster center component of the innovated data matrix (pre-multiplied by the precision matrix).
  - `noise_now`: A \( p \times n \) matrix representing the Gaussian noise component of the data matrix (pre-multiplied by the precision matrix).
  - `Omega_diag_hat`: A \( p \)-dimensional vector of the diagonal entries of the precision matrix.
  - `cluster_est_prev`: An array of positive integers of length \(n\), representing the previous cluster assignments (e.g., `[1 2 1 2 3 4 2]`).

- **Outputs:**
  - `s_hat`: A Boolean vector of length \(p\), where `true` indicates the variable is selected.
```matlab
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
```

### select_variable_ISEE_clean
For Algorithm 3. 
- **Inputs:**
  - `mean_now`: A \( p \times n \) matrix representing the cluster center component of the innovated data matrix (pre-multiplied by the precision matrix), where \(p\) is the data dimension and \(n\) is the sample size.
  - `noise_now`: A \( p \times n \) matrix representing the Gaussian noise component of the data matrix (pre-multiplied by the precision matrix).
  - `Omega_diag_hat`: A vector of length \(p\), containing the diagonal entries of the precision matrix.
  - `cluster_est_prev`: An array of positive integers of length \(n\), representing previous cluster assignments (e.g., `[1 2 1 2 3 4 2]`).

- **Outputs:**
  - `s_hat`: A Boolean vector of length \(p\), where `true` indicates that the corresponding variable is selected.
```matlab
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
```

## SDP clustering

### get_cov_small
- **Inputs:**
  - `x`: A data matrix of shape \( p \times n \), where \(p\) is the data dimension and \(n\) is the sample size.
  - `cluster_est_now`: An array of positive integers of length \(n\), representing current cluster assignments (e.g., `[1 2 1 2 3 4 2]`).
  - `s_hat`: A Boolean vector of length \(p\), where `true` indicates that the corresponding variable is selected.

- **Outputs:**
  - `Sigma_hat_s_hat_now`: The estimated covariance matrix for the selected variables (those indicated by `s_hat`).
```matlab
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
```

### cluster_SDP_noniso
Performs the clustering step of the ISEE-based iterative method. Takes `s_hat` as input, making it applicable to both types of variable selection (Algorithm 6 and 3).

- **Inputs:**
  - `x`: A data matrix of shape \( p \times n \), where \(p\) is the data dimension and \(n\) is the sample size.
  - `K`: A positive integer indicating the number of clusters.
  - `mean_now`: A \( p \times n \) matrix representing the cluster center component of the innovated data matrix (pre-multiplied by the precision matrix).
  - `noise_now`: A \( p \times n \) matrix representing the Gaussian noise component of the data matrix (pre-multiplied by the precision matrix).
  - `cluster_est_prev`: An array of positive integers of length \(n\), representing the previous cluster assignment (e.g., `[1 2 1 2 3 4 2]`).

- **Outputs:**
  - `cluster_est_new`: A length-\(n\) array of updated cluster assignments.
  - `obj_sdp`: A scalar value representing the SDP objective for the current clustering.
  - `obj_lik`: A scalar value representing the likelihood objective for the current clustering.
```matlab
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
```

### ISEE_kmeans_noisy_onestep
One iteration of Algorithm 6.
- **Inputs:**
  - `x`: A data matrix of shape \( p \times n \), where \(p\) is the data dimension and \(n\) is the sample size.
  - `K`: A positive integer specifying the number of clusters.
  - `cluster_est_prev`: An array of positive integers of length \(n\), representing the cluster assignments from the previous step (e.g., `[1 2 1 2 3 4 2]`).
  - `is_parallel`: A Boolean flag indicating whether parallel computing in MATLAB is used.

- **Outputs:**
  - `cluster_est_new`: An array of positive integers of length \(n\), representing the updated cluster assignments (e.g., `[1 2 1 2 3 4 2]`).
  - `obj_sdp`: A scalar representing the SDP objective value.
  - `obj_lik`: A scalar representing the likelihood objective value.
```matlab
  function [cluster_est_new, obj_sdp, obj_lik] = ISEE_kmeans_noisy_onestep(x, K, cluster_est_prev, is_parallel)
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
```

### ISEE_kmeans_noisy
End-user function for Algorithm 6
- **Inputs:**
  - `x`: Data matrix of shape \( p \times n \).
  - `k`: Positive integer specifying the number of clusters.
  - `n_iter`: Maximum number of iterations for the algorithm.
  - `is_parallel`: Boolean flag indicating whether to use parallel execution.
  - `loop_detect_start`: Iteration number at which to start detecting convergence loops.
  - `window_size`: Number of recent steps used to assess stagnation.
  - `min_delta`: Minimum relative improvement required to avoid early stopping.

- **Outputs:**
  - `cluster_est`: A length-\(n\) array of positive integers representing the estimated cluster assignments (e.g., `[1 2 1 2 1 2 2]`).

```matlab
function cluster_estimate = ISEE_kmeans_noisy(x, k, n_iter, is_parallel, loop_detect_start, window_size, min_delta)
    % Initialize tracking vectors
    obj_sdp = nan(1, n_iter);
    obj_lik = nan(1, n_iter);

    % Initial cluster assignment using spectral clustering
    cluster_estimate = cluster_spectral(x, k);

    for iter = 1:n_iter
        % One step of ISEE-based k-means refinement

        [cluster_estimate, s_hat,  obj_sdp(iter), obj_lik(iter)]  = ISEE_kmeans_noisy_onestep(x, k, cluster_estimate, is_parallel);
        fprintf('Iteration %d | SDP obj: %.4f | Likelihood obj: %.4f\n', iter, obj_sdp(iter), obj_lik(iter));

        % Compute objective values

        
        % Early stopping condition
        is_stop = decide_stop(obj_sdp, obj_lik, loop_detect_start, window_size, min_delta);
        if is_stop
            break;
        end
    end
end
```

### ISEE_kmeans_clean_onestep
One step of Algorithm 3

- **Inputs:**
  - `x`: A data matrix of shape \( p \times n \), where \(p\) is the data dimension and \(n\) is the sample size.
  - `K`: A positive integer specifying the number of clusters.
  - `cluster_est_prev`: A length-\(n\) array of positive integers representing the previous cluster assignment (e.g., `[1 2 1 2 3 4 2]`).
  - `is_parallel`: A Boolean flag indicating whether parallel computing in MATLAB is enabled.

- **Outputs:**
  - `cluster_est_new`: A length-\(n\) array of positive integers representing the updated cluster assignment (e.g., `[1 2 1 2 3 4 2]`).
  - `s_hat`: A Boolean vector of length \(p\), indicating which variables were selected.
  - `obj_sdp`: A scalar representing the SDP objective value.
  - `obj_lik`: A scalar representing the likelihood objective value.
```matlab
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
```

### ISEE_kmeans_clean
End-user function for Algorithm 3

- **Inputs:**
  - `x`: Data matrix of shape \( p \times n \).
  - `k`: Positive integer specifying the number of clusters.
  - `n_iter`: Maximum number of iterations for the algorithm.
  - `is_parallel`: Boolean flag indicating whether to use parallel execution.
  - `loop_detect_start`: Iteration number at which to start detecting convergence loops.
  - `window_size`: Number of recent steps used to assess stagnation.
  - `min_delta`: Minimum relative improvement required to avoid early stopping.

- **Output:**
  - `cluster_estimate`: A \( 1 \times n \) array representing the final cluster assignments.

```matlab
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
```

## Algorithm - stopping criterion

### get_sdp_objective
Computes the SDP objective 
- **Inputs:**
  - `X`: A \( p \times n \) data matrix, typically a truncated matrix where \(p = |S|\) for some selected subset of variables \(S\).
  - `G`: A \( 1 \times n \) vector of cluster labels, with entries in \(\{1, \dots, K\}\).
- **Output:**
  - `obj`: SDP objective.

```matlab
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
```

### get_likelihood_objective
- **Inputs:**
  - `X`: A \( p \times n \) data matrix, typically a truncated matrix where \(p = |S|\), the number of selected variables.
  - `G`: A \( 1 \times n \) vector of cluster labels with values in \(\{1, \dots, K\}\).

- **Output:**
  - `obj`: A scalar value representing the profile likelihood objective.

```matlab
sdp_obj = get_sdp_objective(X, G);      % reuse core SDP component
frob_norm_sq = norm(X, 'fro')^2;
obj = sdp_obj - 2 * frob_norm_sq;
end
```


### detect_relative_change
Computes the relative change in the objective value between the last two iterations. This function is typically used in optimization algorithms to monitor convergence. It calculates the relative difference between the two most recent objective values. A small relative change suggests that the algorithm is approaching convergence. 
If the vector contains NaNs, the computation is based on the last two values before the first NaN.
If fewer than two valid values exist, returns Inf.

- **Inputs:**
  - `obj_val_vec`: A numeric vector of objective values over iterations. Must have length \(\geq 2\).

- **Output:**
  - `relative_change`: A scalar representing the relative change between the last two objective values.


```matlab
function is_stuck = detect_relative_change(obj_val_vec, detect_start, min_delta)


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
```




### detect_loop
Detects convergence plateau based on recent objective values.  Resembles keras.callbacks.EarlyStopping (https://keras.io/api/callbacks/early_stopping/), incorporating the min_delta parameter. In our setting, a window of iterations plays the role of epochs in deep learning.
If the last window_size iterations show no improvement over the global optimum so far, return the flag is_loop.
- **Inputs:**
  - `obj_val_vec`: A vector of objective values over iterations. May contain `NaN` values.
  - `loop_detect_start`: An integer indicating the number of initial steps to skip before detecting loops.
  - `window_size`: Number of recent steps to consider for loop detection.
  - `min_delta`: Minimum required relative improvement (in percent) to avoid detection of a loop.

- **Output:**
  - `is_loop`: A logical flag (`true` or `false`). Returns `true` if no significant improvement is detected within the recent window.

```matlab
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
```


### decide_stop

```matlab
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
```

# Simulations - data generator
## get_precision_band
- **Inputs:**
  - `p`: Dimension of the matrix.
  - `precision_sparsity`: Total number of nonzero diagonals. Must be an even number (e.g., 2, 4, 6, ...).
  - `conditional_correlation`: Decay factor applied to off-diagonal entries.

- **Output:**
  - `precision_matrix`: A \( p \times p \) symmetric precision matrix.

```matlab
function precision_matrix = get_precision_band(p, precision_sparsity, conditional_correlation)
% get_precision_band - Constructs a banded symmetric precision matrix with geometric decay
%                      using spdiags, assuming identity base and symmetric off-diagonal decay.
%


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
```


## generate_gaussian_data
Generate two-class Gaussian data with structured precision matrix
- **Inputs:**
  - `n`: Total number of samples.
  - `p`: Number of variables.
  - `model`: Covariance model type; currently only `'AR'` (autoregressive) and chain45 and chain20
  - `seed`: Random seed for reproducibility.
  - `cluster_1_size`: Proportion of samples in class 1 (e.g., `0.5`).

- **Outputs:**
  - `X`: An \( n \times p \) data matrix of generated samples.
  - `y`: An \( n \times 1 \) vector of class labels (1 or 2).
  - `Omega_star`: A \( p \times p \) precision matrix.
  - `beta_star`: A sparse discriminant vector.


```matlab
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
        case 'chain20'
            Omega_star = get_precision_band(p, 2, 0.20);
            Sigma = inv(Omega_star);
        otherwise
            error('Model must be ''chain45'' or ''chain20'' or ''AR''.');
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
```



## Simulation - auxiliary
### get_bicluster_accuracy
Computes accuracy for a two-cluster assignment.
```matlab
acc = GET_CLUSTER_ACCURACY(cluster_est, cluster_true) returns the proportion of correctly assigned labels, accounting for label switching.
Inputs
cluster_est: n array of 1 and 2
cluster_true: n array of 1 and 2
Outputs
acc: ratio of correctly clustered observations

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
```

## Simulation - step-level evaluation

### ISEE_kmeans_clean_simul
```matlab
ISEE_kmeans_clean - Runs iterative clustering with early stopping and logs results to SQLite DB

function cluster_estimate = ISEE_kmeans_clean_simul(x, k, n_iter, is_parallel, loop_detect_start, window_size, min_delta, db_dir, table_name, rep, model, sep, cluster_true)
% 

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
```



