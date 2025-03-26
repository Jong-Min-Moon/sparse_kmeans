classdef sdp_kmeans_bandit_simul  < sdp_kmeans_bandit 


    methods
        function obj = iterative_kmeans(data_object, number_cluster, omega_sparsity)
            obj.X = data_object.data;
            obj.K = number_cluster;
            obj.omega_sparsity = omega_sparsity;
            obj.init_method = "none";

            obj.n = size(obj.X, 2);
            obj.p = size(obj.X, 1);

            C = 0.5;
            obj.cutoff = log(1 / C) / log((1 + C) / C);


            obj.n_iter = NaN;
            obj.set_bayesian_parameters();
        end

        function fit_predict(obj, n_iter, cluster_true)
            obj.n_iter = n_iter;
            obj.initialize_cluster_est();
            obj.initialize_saving_matrix()

            for i = 1:n_iter
                variable_subset_now = obj.choose();
                obj.entries_survived(i, :) = variable_subset_now;
                disp(['Iteration ', num2str(i), ' - arms pulled: ', mat2str(find(variable_subset_now))]);
                disp(['number of arms pulled: ', mat2str(sum(variable_subset_now))]);
                reward_now = obj.reward(variable_subset_now, i);
                obj.update(variable_subset_now, reward_now);

                obj.evaluate_accuracy(cluster_true, i);
            end
            
            %final clustering
            final_selection = obj.signal_entry_est;
            obj.entries_survived(obj.n_iter + 1, :) = final_selection;
            X_sub_final = obj.X(final_selection, :);
            kmeans_learner = sdp_kmeans(X_sub_final, obj.K);
            obj.cluster_est_dict(obj.n_iter + 1) = cluster_est(kmeans_learner.fit_predict());
            obj.evaluate_accuracy(cluster_true, obj.n_iter + 1);
        end
        
        function evaluate_accuracy(obj, cluster_true, iter)
            cluster_est_now = obj.cluster_est_dict(iter);
            obj.acc_dict(iter) = cluster_est_now.evaluate_accuracy(cluster_true);
            obj.acc_dict(iter)
        end % end of method evaluate_accuracy

        function initialize_saving_matrix(obj)
            obj.x_tilde_est       = zeros(obj.p, obj.n, obj.n_iter);
            obj.omega_est_time    = zeros(obj.n_iter, 1);
            obj.sdp_solve_time    = zeros(obj.n_iter, 1);
            obj.entries_survived  = zeros(obj.n_iter+1, obj.p);
            obj.obj_val_prim      = zeros(obj.n_iter, 1);
            obj.obj_val_dual      = zeros(obj.n_iter, 1);
            obj.obj_val_original  = zeros(obj.n_iter, 1);
        end

        function cluster_est_obj = fetch_cluster_est(obj,iter)
            cluster_est_obj = obj.cluster_est_dict(iter);
        end

        function database_subtable = get_database_subtable(obj, rep, Delta, rho, support, cluster_true, Omega)
            s = length(support);
            current_time = get_current_time();
            [true_pos_vec, false_pos_vec, false_neg_vec, survived_indices] = obj.evaluate_discovery(support);
            [diff_x_tilde_fro, diff_x_tilde_op, diff_x_tilde_ellone] = obj.evaluate_innovation_est(Omega);
            %fprintf( strcat( "acc =", join(repelem("%f ", length(acc_vec))), "\n"),  acc_vec );
            
            
            cluster_string_dict = obj.get_cluster_string_dict();
            
            values(acc_dict)
            values(cluster_string_dict)
    
            n_row = int32(obj.n_iter);
            database_subtable = table(...
                repelem(rep, n_row+1)',...                      % 01 replication number
                (0:n_row)',...                                  % 02 step iteration number
                repelem(Delta, n_row+1)',...                    % 03 separation
                repelem(obj.data_object.dimension, n_row+1)',... % 04 data dimension
                repelem(rho, n_row+1)',...                      % 05 conditional correlation
                repelem(s, n_row+1)',...                        % 06 sparsity
                ...
                [false; obj.stop_decider.stop_history{1:n_row, "original"}],... %07
                [false; obj.stop_decider.stop_history{1:n_row, "sdp"}],...      %08
                [false; obj.stop_decider.stop_history{1:n_row, "loop"}],...     %09
                cell2mat(values(acc_dict))',...                                     % 10 accuracy
                ...
                [0; obj.obj_val_prim(1:n_row)],...               % 11 objective function value (relaxed, primal)
                [0; obj.obj_val_dual(1:n_row)],...               % 12 objective function value (relaxed, dual)
                [0; obj.obj_val_original(1:n_row)],...           % 13 objective function value (original)
                ...
                [0; true_pos_vec],...                           % 14 true positive
                [0; false_pos_vec],...                          % 15 false positive
                [0; false_neg_vec],...                          % 16 false negative
                ...
                [0; diff_x_tilde_fro],...                       % 13 estimation error of the innovated data, in Frobenius norm
                [0; diff_x_tilde_op],...                        % 14 estimation error of the innovated data, in operator norm
                [0; diff_x_tilde_ellone],...                    % 15 estimation error of the innovated data, in \ell_1 norm
                [0; obj.omega_est_time(1:n_row)],...             % 16 timing for estimating the precision matrix
                [0; obj.sdp_solve_time(1:n_row)], ...            % 17 timing elapsed for solving the SDP
                repelem(current_time, n_row+1)', ...            % 18 timestamp
                [""; survived_indices],...                      % 19 indices of survived entry
                string(values(cluster_string_dict))',...                          % 20 clustering information
                'VariableNames', ...
                ...  1      2       3      4      5        6         
                ["rep", "iter", "sep", "dim", "rho", "sparsity", ...
                ...
                "stop_og", "stop_sdp", "stop_loop", ...
                ...  7        8           9             10             
                 "acc", "obj_prim", "obj_dual", "obj_original", ...
                ...11               12
                 "true_pos", "false_pos",  "false_neg"...
                ...       13              14                    15
                 "diff_x_tilde_fro", "diff_x_tilde_op", "diff_x_tilde_ellone", ...
                ...  16          17           18
                 "time_est", "time_SDP", "jobdate", ...
                ...      19              20            
                "survived_indices", "cluster_est"
                ]);
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
        end
    
    end % end of method
end % end of class