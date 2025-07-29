classdef sdp_kmeans_iter_knowncov_ifpca < sdp_kmeans_iter_knowncov
%% sdp_kmeans_iter_knowncov_ifpca
% @export
       methods
    
           function cluster_est = get_cluster_initial(obj, X, K)
            cluster_est = ifpca(X, K);
        end
       end
end
