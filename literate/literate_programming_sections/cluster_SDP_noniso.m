function cluster_est_new = cluster_SDP_noniso(x, K, mean_now, noise_now, cluster_est_prev, s_hat)
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
