classdef sdp_kmeans_iter_knowncov_SL < sdp_kmeans_iter_knowncov
%% sdp_kmeans_iter_knowncov_SL
% @export
       methods
    
        function obj = sdp_kmeans_iter_knowncov_SL(X, K)
            obj = obj@sdp_kmeans_iter_knowncov(X, K);
        end        
        function cluster_est = get_cluster(obj, X, K)
            cluster_est = get_cluster_by_sdp_SL(X, K);
        end
       end
end
