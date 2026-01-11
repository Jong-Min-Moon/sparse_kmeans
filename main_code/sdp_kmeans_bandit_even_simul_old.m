classdef sdp_kmeans_bandit_even_simul_old  < sdp_kmeans_bandit_simul 
%% sdp_kmeans_bandit_even_simul_old
% @export
    methods
        function obj = sdp_kmeans_bandit_even_simul_old(X, number_cluster)
            % Call the superclass constructor first
            % This initializes X, K, n, p, cutoff, and n_iter properties from the superclass
            obj = obj@sdp_kmeans_bandit_simul(X, number_cluster);
            
        end
        
    
        function reward_vec = reward(obj, variable_subset, iter)
            % Use only selected variables
            num_selected_features = sum(variable_subset);
            if num_selected_features == 0
                % If no variables selected, no reward can be computed for features.
                % All rewards are 0, and we skip clustering.
                reward_vec = zeros(1, obj.p); 
                return; % Exit early
            end
            X_sub = obj.X(variable_subset, :);
            % Perform clustering on X_sub (original selected data)
            obj.cluster_est_dict(iter) = obj.get_cluster(X_sub, obj.K);
            
            % --- 1. Add standard normal matrix to X_sub (after clustering) ---
            % Dimensions of X_sub are (num_selected_features x num_data_points)
            X_sub_noisy = X_sub + randn(num_selected_features, obj.n); % Use obj.n for num_data_points
            % Get cluster estimates
            cluster_labels = obj.cluster_est_dict(iter).cluster_info_vec;
            % --- 2. Compute cluster_norm for each feature (VECTORIZED) ---
            A_double = get_assignment_matrix(obj.n, obj.K, cluster_labels);
            % Calculate the sum of each feature's values within each cluster
            % Resulting matrix 'cluster_sums' will be (num_selected_features x K)
            cluster_sums = X_sub_noisy * A_double; 
            cluster_sums_sq = cluster_sums.^2;
            cluster_norm = sum(cluster_sums_sq, 2)'; 
            % --- 3. Define reward_vec by thresholding cluster_norm (VECTORIZED) ---
            % Calculate the threshold
            q = obj.n * (log(obj.n) + log(obj.p)) / obj.K;
            threshold = sqrt(obj.n * q) + q;
            
            % Initialize reward_vec (full length of original features, p)
            reward_vec = zeros(1, obj.p); 
            
            % Get original indices of selected variables
            idx = find(variable_subset); 
            
            % Directly assign the thresholded values using vectorized indexed assignment
            reward_vec(idx) = cluster_norm > threshold; 
            
        end % end of method reward            
     
 
    end % end of methods
end
