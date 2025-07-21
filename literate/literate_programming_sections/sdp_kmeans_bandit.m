function obj = sdp_kmeans_bandit(X, K)
%% sdp_kmeans_bandit
% @export
 classdef sdp_kmeans_bandit < handle
    properties
        X           % Data matrix (d x n)
        K           % Number of clusters
        n           % Number of data points
        p           % Data dimension
        cutoff      % Threshold for variable inclusion
        alpha       % Alpha parameters of Beta prior
        beta        % Beta parameters of Beta prior
        pi
        acc_dict
        cluster_est_dict
        signal_entry_est
        n_iter
        x_tilde_est       
        omega_est_time    
        sdp_solve_time    
        entries_survived  
        obj_val_prim     
        obj_val_dual      
        obj_val_original  
    end
    methods
            % Constructor
            if nargin < 2
                error('Two input arguments required: data matrix X and number of clusters K.');
            end
            if ~ismatrix(X) || ~isnumeric(X)
                error('Input X must be a numeric matrix.');
            end
            if ~isscalar(K) || K <= 1 || K ~= floor(K)
                error('Number of clusters K must be an integer greater than 1.');
            end
            obj.X = X;
            obj.K = K;
            obj.n = size(X, 2);
            obj.p = size(X, 1);
            C = 0.5;
            obj.cutoff = log(1 / C) / log((1 + C) / C);
            obj.n_iter = NaN;
            
            
            
        end
        
        function set_bayesian_parameters(obj)            
            obj.alpha = ones(1, obj.p);
            obj.beta = repmat(1, 1, obj.p);
            obj.pi = obj.alpha ./ (obj.alpha + obj.beta);
        end
        function fit_predict(obj, n_iter)
            obj.n_iter = n_iter;
            obj.set_bayesian_parameters();
            obj.initialize_cluster_est();
            fprintf("initialization done")
            for i = 1:n_iter
                variable_subset_now = obj.choose();
                disp(['Iteration ', num2str(i), ' - arms pulled: ', mat2str(find(variable_subset_now))]);
                disp(['number of arms pulled: ', mat2str(sum(variable_subset_now))]);
                reward_now = obj.reward(variable_subset_now, i);
                obj.update(variable_subset_now, reward_now);
            end
            %final clustering
            final_selection = obj.signal_entry_est;
            X_sub_final = obj.X(final_selection, :);
            kmeans_learner = sdp_kmeans(X_sub_final, obj.K);
            obj.cluster_est_dict(obj.n_iter + 1) = cluster_est(kmeans_learner.fit_predict());
            
        end
        
        function initialize_cluster_est(obj)
            cluster_est_dummy   = cluster_est( repelem(1,obj.n) );
            obj.cluster_est_dict = repelem(cluster_est_dummy, obj.n_iter+1); %dummy
            obj.acc_dict = containers.Map(1:(obj.n_iter+1), repelem(0, obj.n_iter+1)); 
        end
        function variable_subset = choose(obj)
            theta = betarnd(obj.alpha, obj.beta);
            variable_subset = theta > obj.cutoff;
        end
        
        function reward_vec = reward(obj, variable_subset, iter)
            % Use only selected variables
            X_sub = obj.X(variable_subset, :);
            kmeans_learner = sdp_kmeans(X_sub, obj.K);
            obj.cluster_est_dict(iter) = cluster_est(kmeans_learner.fit_predict()); 
            % Assume K = 2
            sample_cluster_1 = X_sub(:, obj.cluster_est_dict(iter).cluster_info_vec == 1);
            sample_cluster_2 = X_sub(:, obj.cluster_est_dict(iter).cluster_info_vec == 2);
            %size(sample_cluster_1, 2)
            %size(sample_cluster_2, 2)
            reward_vec = zeros(1, obj.p);
            idx = find(variable_subset);
            % only calculate the p-values for selected variables
            for j = 1:length(idx)
                i = idx(j);
                p_val =  permutationTest( ...
                    sample_cluster_1(j, :), ...
                    sample_cluster_2(j, :), ...
                    100 ...
                ); % 
                reward_vec(i) = p_val < 0.01;
            end
            
     
        end % end of method reward
        function update(obj, variable_subset, reward_vec)
            obj.alpha(variable_subset) = obj.alpha(variable_subset) + reward_vec(variable_subset);
            obj.beta(variable_subset) = obj.beta(variable_subset) + (1 - reward_vec(variable_subset));
            obj.pi = obj.alpha ./ (obj.alpha + obj.beta); 
            obj.signal_entry_est = obj.pi>0.5;
        end % end of method update    
    end % end of methods
end
%% 
