function cluster_estimate = ISEE_kmeans_clean(x, k, n_iter, is_parallel)
%% ISEE_kmeans_clean
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
%initialization
    cluster_estimate = cluster_spectral(x, k);
    for iter = 1:n_iter
        cluster_estimate = ISEE_kmeans_clean_onestep(x, k, cluster_estimate, is_parallel);
    end
end
