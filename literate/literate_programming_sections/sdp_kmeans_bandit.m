classdef sdp_kmeans_bandit < handle
%% sdp_kmeans_bandit
% @export
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
        signal_entry_est
        n_iter
        cluster_est
        x_tilde_est       
        omega_est_time    
        sdp_solve_time    
        entries_survived  
        obj_val_prim     
        obj_val_dual      
        obj_val_original  
    end
    methods
        function obj = sdp_kmeans_bandit(X, K)
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
            tic; % Start timing for the entire fit_predict method
            obj.n_iter = n_iter;
            obj.set_bayesian_parameters();
            obj.initialize_cluster_est();
            fprintf("initialization done\n")
            for i = 1:n_iter
                variable_subset_now = obj.choose();
                %disp(['Iteration ', num2str(i), ' - arms pulled: ', mat2str(find(variable_subset_now)), '\n']);
                disp(['number of arms pulled: ', mat2str(sum(variable_subset_now)), '\n']);
                reward_now = obj.reward(variable_subset_now, i);
                obj.update(variable_subset_now, reward_now);
            end
            %final clustering
            final_selection = obj.signal_entry_est;
            X_sub_final = obj.X(final_selection, :);
            obj.cluster_est = obj.get_cluster(X_sub_final, obj.K);
            % ... all existing code ...
        total_fit_predict_time = toc; % End timing for the entire fit_predict method            
        fprintf('Total fit_predict time: %.4f seconds\n', total_fit_predict_time);
        end
  
        function cluster_est = get_cluster(obj, X, K) % inherit this class and change this part to try simpler clustering methods
            cluster_est = get_cluster_by_sdp(X, K);
        end
        function initialize_cluster_est(obj)
              obj.acc_dict = containers.Map(1:(obj.n_iter+1), repelem(0, obj.n_iter+1)); 
        end
        function variable_subset = choose(obj)
            theta = betarnd(obj.alpha, obj.beta);
            variable_subset = theta > obj.cutoff;
        end
        
        function reward_vec = reward(obj, variable_subset, iter)
            % Use only selected variables
            X_sub = obj.X(variable_subset, :);
            obj.cluster_est  = obj.get_cluster(X_sub, obj.K);
            % Assume K = 2
            sample_cluster_1 = X_sub(:, obj.cluster_est == 1);
            sample_cluster_2 = X_sub(:, obj.cluster_est == 2);
            %size(sample_cluster_1, 2)
            %size(sample_cluster_2, 2)
            reward_vec = zeros(1, obj.p);
            idx = find(variable_subset);
            % only calculate the p-values for selected variables
            for j = 1:length(idx)
                i = idx(j);
                pval =  permutationTest( ...
                    sample_cluster_1(j, :), ...
                    sample_cluster_2(j, :), ...
                    100 ...
                ); % 
                reward_vec(i) = pval < 0.01;
            end
            disp(['number of rewarded pulls: ', mat2str(sum(reward_vec))]);
            
     
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
%% 
% 
% 
% 
