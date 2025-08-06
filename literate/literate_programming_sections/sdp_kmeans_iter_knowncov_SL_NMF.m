classdef sdp_kmeans_iter_knowncov_SL_NMF < sdp_kmeans_iter_knowncov
%% sdp_kmeans_iter_knowncov_SL_NMF
% @export
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
