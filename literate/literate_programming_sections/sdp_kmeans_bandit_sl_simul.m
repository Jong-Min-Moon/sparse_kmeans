classdef sdp_kmeans_bandit_sl_simul  < sdp_kmeans_bandit_simul 
%% sdp_kmeans_bandit_sl_simul
% @export
    methods
        function obj = sdp_kmeans_bandit_sl_simul(X, number_cluster)
            % Call the superclass constructor first
            % This initializes X, K, n, p, cutoff, and n_iter properties from the superclass
            obj = obj@sdp_kmeans_bandit_simul(X, number_cluster);
            
        end
        
        function cluster_est = get_cluster(obj, X, K) % inherit this class and change this part to try simpler clustering methods
            cluster_est = get_cluster_by_sdp_SL_NMF(X, K);
        end    
 
        function reward_vec = reward(obj, variable_subset, iter)
            % Use only selected variables
            X_sub = obj.X(variable_subset, :);
            n_selected_feature = size(variable_subset,2);
            obj.cluster_est  = obj.get_cluster(X_sub, obj.K);
                            n_g1_now = sum( obj.cluster_est == 1);
                n_g2_now = obj.n-n_g1_now;
            % Assume K = 2
            sample_cluster_1 = X_sub(:, obj.cluster_est == 1);
            sample_cluster_2 = X_sub(:, obj.cluster_est == 2);
                 x_bar_g1 = mean(sample_cluster_1, 2);  
                  x_bar_g2 = mean(sample_cluster_2, 2);
            % thresholding
            reward_vec = zeros(1, obj.p);
            idx = find(variable_subset);
            abs_diff = abs(x_bar_g1 - x_bar_g2) * sqrt( n_g1_now*n_g2_now/obj.n );
                cutoff_now =   sqrt(2 * log(obj.p) );
                reward_vec(idx) = abs_diff > cutoff_now;
                n_selected_features = sum(reward_vec);
                fprintf("%i entries got a reward \n\n",n_selected_features)
              
       
            
     
        end % end of method reward
     
 
    end % end of methods
end
%% 
