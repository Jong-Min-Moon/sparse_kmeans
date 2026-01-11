classdef sdp_kmeans_bandit_thinning_nmf_simul  < sdp_kmeans_bandit_thinning_simul 
%% sdp_kmeans_bandit_thinning_nmf_simul
% @export
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
