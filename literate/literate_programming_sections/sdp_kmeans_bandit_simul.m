classdef sdp_kmeans_bandit_simul  < sdp_kmeans_bandit 
%% sdp_kmeans_bandit_simul
% @export
    methods
        function obj = sdp_kmeans_bandit_simul(X, number_cluster)
            % Call the superclass constructor first
            % This initializes X, K, n, p, cutoff, and n_iter properties from the superclass
            
            obj = obj@sdp_kmeans_bandit(X, number_cluster);
            
        end
        function fit_predict(obj, n_iter, cluster_true)
            obj.n_iter = n_iter;
            obj.set_bayesian_parameters();
            obj.initialize_cluster_est();
            obj.initialize_saving_matrix()
            for i = 1:n_iter
                variable_subset_now = obj.choose();
                obj.entries_survived(i, :) = variable_subset_now;
                arms_pulled = mat2str(find(variable_subset_now));
                disp(['Iteration ', num2str(i), ' - arms pulled: ', arms_pulled(1: min(20, size(arms_pulled,2)))]);
                disp(['number of arms pulled: ', mat2str(sum(variable_subset_now))]);
                reward_now = obj.reward(variable_subset_now, i);
                obj.update(variable_subset_now, reward_now);
                obj.evaluate_accuracy(obj.cluster_est, cluster_true, i);
            end
            
            %final clustering
            final_selection = obj.signal_entry_est;
 
            X_sub_final = obj.X(final_selection, :);
            obj.cluster_est = obj.get_cluster(X_sub_final, obj.K);
            obj.evaluate_accuracy(obj.cluster_est, cluster_true, obj.n_iter + 1);
        end
        
        function evaluate_accuracy(obj, cluster_est, cluster_true, iter)
             obj.acc_dict(iter) = get_bicluster_accuracy(cluster_est, cluster_true);
            obj.acc_dict(iter)
        end % end of method evaluate_accuracy
        function initialize_saving_matrix(obj)
             obj.omega_est_time    = zeros(obj.n_iter, 1);
            obj.sdp_solve_time    = zeros(obj.n_iter, 1);
             obj.obj_val_prim      = zeros(obj.n_iter, 1);
            obj.obj_val_dual      = zeros(obj.n_iter, 1);
            obj.obj_val_original  = zeros(obj.n_iter, 1);
        end
      
        function database_subtable = get_database_subtable(obj, rep, Delta, support)
            s = length(support);
            current_time = get_current_time();
            [true_pos_vec, false_pos_vec, false_neg_vec, ~] = obj.evaluate_discovery(support);
            %fprintf( strcat( "acc =", join(repelem("%f ", length(acc_vec))), "\n"),  acc_vec );
            
            
             
            %values(obj.acc_dict);
            %values(cluster_string_dict);
             
            n_row = int32(obj.n_iter);
            database_subtable = table(...
                repelem(rep, n_row+1)',...                      % 01 replication number
                (1:(n_row+1))',...                              % 02 step iteration number
                repelem(Delta, n_row+1)',...                    % 03 separation
                repelem(obj.p, n_row+1)',...                    % 04 data dimension
                repelem(obj.n, n_row+1)',...                      % 05 sample size
                repelem(s, n_row+1)',...                        % 06 model
                ...
                cell2mat(values(obj.acc_dict))',...             % 07 accuracy
                ...
                repelem(0, n_row+1)',...               % 8 sdp objective function value  
                repelem(0, n_row+1)',...               % 9 likelihood value
                ...
                [0; true_pos_vec],...                           % 10 true positive
                [0; false_pos_vec],...                          % 11 false positive
                [0; false_neg_vec],...                          % 12 false negative
                ...
                repelem(current_time, n_row+1)', ...            % 13 timestamp
                'VariableNames', ...
                ...  %1      2       3      4      5        6         
                ["rep", "iter", "sep", "dim", "n", "model", ...
                ...  %7        8           9                       
                 "acc", "obj_sdp", "obj_lik",  ...
                ... % 10          11            12
                 "true_pos", "false_pos",  "false_neg",...
                ...  13
                     "jobdate"]);
        end % end of get_database_subtable
 
        function [true_pos_vec, false_pos_vec, false_neg_vec , survived_indices] = evaluate_discovery(obj, support)
            true_pos_vec  = zeros(obj.n_iter, 1);
            false_pos_vec = zeros(obj.n_iter, 1);
            false_neg_vec = zeros(obj.n_iter, 1);
            survived_indices = strings(obj.n_iter, 1);
            for i = 1:obj.n_iter
                positive_vec = obj.entries_survived(i,:);
                true_pos_vec(i)  = sum(positive_vec(support));
                false_pos_vec(i) = sum(positive_vec) - true_pos_vec(i);
    
                negative_vec = ~positive_vec;
                false_neg_vec(i) = sum(negative_vec(support));
                survived_indices(i) = get_num2str_with_mark( find(positive_vec), ',');
            end
        end % end of evaluate_discovery
  
    
    end % end of method
end % end of class
