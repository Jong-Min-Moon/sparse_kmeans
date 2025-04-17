function cluster_est_new = cluster_SDP_noniso(x, K, mean_now, noise_now, cluster_est_prev, s_hat)
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
    %estimate sigma hat s
    n = size(x,2);
    Sigma_hat_s_hat_now = get_cov_small(x, cluster_est_prev, s_hat);
    x_tilde_now = mean_now + noise_now;
    x_tilde_now_s  = x_tilde_now(s_hat,:);  
    Z_now = kmeans_sdp_pengwei( x_tilde_now_s' * Sigma_hat_s_hat_now * x_tilde_now_s/ n, K);
    % final thresholding
    [U_sdp,~,~] = svd(Z_now);
    U_top_k = U_sdp(:,1:K);
    [cluster_est_new,C] = kmeans(U_top_k,K);  % label
    cluster_est_new = cluster_est_new';    
end
%% 
%% 
