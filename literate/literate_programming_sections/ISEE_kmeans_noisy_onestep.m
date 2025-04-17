function cluster_est_new = ISEE_kmeans_noisy_onestep(x, K, cluster_est_prev, is_parallel)
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
