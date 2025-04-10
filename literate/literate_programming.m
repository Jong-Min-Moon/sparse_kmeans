function z = dummy(x,y)
%% DUMMY 
    z = x+y;
end
%% 
% 
%% Basic functions
% 
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
function [mean_now, noise_now, Omega_diag_hat] = ISEE_bicluster_parallel(x, cluster_est_now)
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
    parfor i = 1 : n_regression
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
    thres = sqrt(2 * log(p) );
    signal_est_now = mean( x_tilde_now(:, cluster_est_prev==1), 2) - mean( x_tilde_now(:, cluster_est_prev==2), 2);
    n_g1_now = sum(cluster_est_prev == 1);
    n_g2_now = sum(cluster_est_prev == 2);
    abs_diff = abs(signal_est_now')./sqrt(Omega_diag_hat) * sqrt( n_g1_now*n_g2_now/n );
    s_hat = abs_diff > thres;
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
    X_g1_now = x(:, (cluster_est ==  1)); 
    X_g2_now = x(:, (cluster_est ==  2)); 
    X_mean_g1_now = mean(X_g1_now, 2);
    X_mean_g2_now = mean(X_g2_now, 2);
    data_py = [(X_g1_now - X_mean_g1_now) (X_g2_now - X_mean_g2_now)]';
    data_filtered = data_py(:,s_hat);
    Sigma_hat_s_hat_now = data_filtered' * data_filtered/(n-1);
end
%% cluster_SDP_noniso
% @export
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
%% 
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
    cluster_est_now = cluster_est_now';   
end
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
function ISEE_kmeans_noisy_onestep(x, K, cluster_est_prev, is_parallel)
%estimation
    if is_parallel
        [mean_now, noise_now, Omega_diag_hat] = ISEE_bicluster_parallel(x, cluster_est_prev);
    else
        [mean_now, noise_now, Omega_diag_hat] = ISEE_bicluster_parallel(x, cluster_est_prev);
    end
%variable selection
    s_hat = select_variable_ISEE_noisy(mean_now, noise_now, Omega_diag_hat, cluster_est_prev);
%clustering
    cluster_est_new = cluster_SDP_noniso(x, K, mean_now, noise_now, cluster_est_prev, s_hat);
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
function cluster_est = ISEE_kmeans_noisy(x, K, n_iter, is_parallel)
%initialization
    cluster_est = cluster_spectral(x, k);
    for iter = 1:n_iter
        cluster_est = ISEE_kmeans_noisy_onestep(x, K, cluster_est, is_parallel);
    end
end