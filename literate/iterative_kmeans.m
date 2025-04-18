classdef iterative_kmeans < handle

properties
    data_object
    number_cluster
    omega_sparsity
    init_method
    x_tilde_est
    omega_est_time
    sdp_solve_time
    entries_survived
    obj_val_prim
    obj_val_dual
    obj_val_original
    cluster_est_dict
    iter_stop
    stop_decider
end

methods
    function ik = iterative_kmeans(data_object, number_cluster, omega_sparsity, init_method)
        ik.data_object = data_object;
        ik.number_cluster = number_cluster;
        ik.omega_sparsity = omega_sparsity;
        ik.init_method = init_method;
    end
    
    function run_single_iter(ik, iter)
        fprintf("\n%i th thresholding\n\n", iter)
        cluster_now = ik.fetch_cluster_est(iter-1);
        %estimation and thresholding
        tic
        [data_innovated_small, data_innovated_big, sample_covariance_small] = ik.data_object.threshold(cluster_now, ik.omega_sparsity);

        ik.omega_est_time(iter) = toc;
        ik.entries_survived(iter,:) = ik.data_object.support;
        ik.x_tilde_est(:,:,iter) = data_innovated_big;
        n_survived = sum(ik.data_object.support);
        fprintf("\n%i entries survived \n",n_survived)
        if n_survived > 0
            fprintf('solving SDP...')
            tic
            [Z_now, obj_val] = kmeans_sdp( data_innovated_small' * sample_covariance_small * data_innovated_small/ ik.data_object.sample_size, ik.data_object.number_cluster);
            cluster_est_vec = sdp_to_cluster(Z_now, ik.data_object.number_cluster);
            ik.sdp_solve_time(iter) = toc;
            ik.obj_val_prim(iter) = obj_val(1);
            ik.obj_val_dual(iter) = obj_val(2);
            ik.obj_val_original(iter) = ik.get_objective_value_original(cluster_est_vec);
            fprintf('took %fs, relaxed dual: %f, original: %f \n', [ik.sdp_solve_time(iter), ik.obj_val_dual(iter), ik.obj_val_original(iter)])
        else
            fprintf('All entries dead. Re-initializing...')
            cluster_est_vec = ik.get_initial_cluster_assign();
        end
        ik.insert_cluster_est(cluster_est_vec, iter);
    end
    
    function [cluster_est_final, iter_stop] = run_iterative_algorithm(ik, max_n_iter, window_size_half, percent_change, run_full, loop_detect_start)
        ik.stop_decider = stopper(max_n_iter, window_size_half, percent_change, loop_detect_start);
        ik.initialize_saving_matrix(max_n_iter)
  
        %initialization
        initial_cluster_assign = ik.get_initial_cluster_assign();
        ik.insert_cluster_est(initial_cluster_assign, 0);
        
        for iter = 1:max_n_iter
            ik.run_single_iter(iter)
            
            % stopping criterion
            criteria_vec = ik.stop_decider.apply_criteria(ik.obj_val_original, ik.obj_val_prim, iter);
            [is_stop, final_iter] = ik.stop_decider.is_stop_by_two(iter)
            if is_stop
                ik.iter_stop = final_iter;
                fprintf("\n final iteration = %i ", ik.iter_stop)
                if ~run_full
                    break 
                end
            end %end of stopping criteria
        end % end one iteration
        cluster_est_final = ik.fetch_cluster_est(ik.iter_stop);
        iter_stop = ik.iter_stop;
    end

    function insert_cluster_est(ik, cluster_info_vec, iter)
        ik.cluster_est_dict(iter+1) = cluster_est(cluster_info_vec); 
    end

    function cluster_est_obj = fetch_cluster_est(ik,iter)
        cluster_est_obj = ik.cluster_est_dict(iter+1);
    end

    function initialize_saving_matrix(ik, max_n_iter)
        p = ik.data_object.dimension;
        n = ik.data_object.sample_size;
        ik.x_tilde_est       = zeros(p, n, max_n_iter);
        ik.omega_est_time    = zeros(max_n_iter, 1);
        ik.sdp_solve_time    = zeros(max_n_iter, 1);
        ik.entries_survived  = zeros(max_n_iter, p);
        ik.obj_val_prim      = zeros(max_n_iter, 1);
        ik.obj_val_dual      = zeros(max_n_iter, 1);
        ik.obj_val_original  = zeros(max_n_iter, 1);

        cluster_est_dummy   = cluster_est( repelem(1,n) );
        ik.cluster_est_dict = repelem(cluster_est_dummy, max_n_iter+1); %dummy
    end

    function initial_cluster_assign = get_initial_cluster_assign(ik)
        if strcmp(ik.init_method, 'spec')
            H_hat = (ik.data_object.data' * ik.data_object.data)/ik.data_object.sample_size;
            [V,D] = eig(H_hat);
            [d,ind] = sort(diag(D), "descend");
            Ds = D(ind,ind);
            Vs = V(:,ind);
            [initial_cluster_assign,~] = kmeans(Vs(:,1), ik.number_cluster);
        elseif strcmp(ik.init_method, "hc")
            Z = linkage(ik.data_object.data', 'ward');
            initial_cluster_assign = cluster(Z, 'Maxclust', ik.number_cluster);
        elseif strcmp(ik.init_method, "sdp")
            [Z_now, obj_val] = kmeans_sdp( ik.data_object.data' * ik.data_object.data/ik.data_object.sample_size, ik.number_cluster);
            initial_cluster_assign = sdp_to_cluster(Z_now, ik.number_cluster);
        end
    end
    


    function database_subtable = get_database_subtable(ik, rep, Delta, rho, support, cluster_true, Omega)
        s = length(support);
        current_time = get_current_time();
        acc_dict = ik.evaluate_accuracy(cluster_true)
        [true_pos_vec, false_pos_vec, false_neg_vec, survived_indices] = ik.evaluate_discovery(support);
        [diff_x_tilde_fro, diff_x_tilde_op, diff_x_tilde_ellone] = ik.evaluate_innovation_est(Omega);
        %fprintf( strcat( "acc =", join(repelem("%f ", length(acc_vec))), "\n"),  acc_vec );
        
        
        cluster_string_dict = ik.get_cluster_string_dict();
        
        values(acc_dict)
        values(cluster_string_dict)

        n_row = int32(ik.iter_stop);
        database_subtable = table(...
            repelem(rep, n_row+1)',...                      % 01 replication number
            (0:n_row)',...                                  % 02 step iteration number
            repelem(Delta, n_row+1)',...                    % 03 separation
            repelem(ik.data_object.dimension, n_row+1)',... % 04 data dimension
            repelem(rho, n_row+1)',...                      % 05 conditional correlation
            repelem(s, n_row+1)',...                        % 06 sparsity
            ...
            [false; ik.stop_decider.stop_history{1:n_row, "original"}],... %07
            [false; ik.stop_decider.stop_history{1:n_row, "sdp"}],...      %08
            [false; ik.stop_decider.stop_history{1:n_row, "loop"}],...     %09
            cell2mat(values(acc_dict))',...                                     % 10 accuracy
            ...
            [0; ik.obj_val_prim(1:n_row)],...               % 11 objective function value (relaxed, primal)
            [0; ik.obj_val_dual(1:n_row)],...               % 12 objective function value (relaxed, dual)
            [0; ik.obj_val_original(1:n_row)],...           % 13 objective function value (original)
            ...
            [0; true_pos_vec],...                           % 14 true positive
            [0; false_pos_vec],...                          % 15 false positive
            [0; false_neg_vec],...                          % 16 false negative
            ...
            [0; diff_x_tilde_fro],...                       % 13 estimation error of the innovated data, in Frobenius norm
            [0; diff_x_tilde_op],...                        % 14 estimation error of the innovated data, in operator norm
            [0; diff_x_tilde_ellone],...                    % 15 estimation error of the innovated data, in \ell_1 norm
            [0; ik.omega_est_time(1:n_row)],...             % 16 timing for estimating the precision matrix
            [0; ik.sdp_solve_time(1:n_row)], ...            % 17 timing elapsed for solving the SDP
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
    end

    

    function acc_dict = evaluate_accuracy(ik, cluster_true)
        acc_dict = containers.Map(0:ik.iter_stop, repelem(0, ik.iter_stop+1));     
        for iter = 0:ik.iter_stop
            cluster_est_now = ik.fetch_cluster_est(iter);
            acc_dict(iter) = cluster_est_now.evaluate_accuracy(cluster_true);
        end
    end

    function cluster_string_vec = get_cluster_string_dict(ik)
        cluster_string_vec = containers.Map(0:ik.iter_stop, repelem("", ik.iter_stop+1));     
        for iter = 0:ik.iter_stop
            cluster_est_now = ik.fetch_cluster_est(iter);
            cluster_string_vec(iter) = cluster_est_now.cluster_info_string;
        end % end of for loop
    end% end of cluster_string_vec   
 
    function [true_pos_vec, false_pos_vec, false_neg_vec , survived_indices] = evaluate_discovery(ik, support)
        true_pos_vec  = zeros(ik.iter_stop, 1);
        false_pos_vec = zeros(ik.iter_stop, 1);
	    false_neg_vec = zeros(ik.iter_stop, 1);
        survived_indices = strings(ik.iter_stop, 1);
        for i = 1:ik.iter_stop
            positive_vec = ik.entries_survived(i,:);
            true_pos_vec(i)  = sum(positive_vec(support));
            false_pos_vec(i) = sum(positive_vec) - true_pos_vec(i);

	        negative_vec = ~positive_vec;
	        false_neg_vec(i) = sum(negative_vec(support));
            survived_indices(i) = get_num2str_with_mark( find(positive_vec), ',');
        end
    end

    function [diff_x_tilde_fro, diff_x_tilde_op, diff_x_tilde_ellone] = evaluate_innovation_est(ik,Omega)
        oracle = Omega * ik.data_object.data;
        diff_x_tilde_fro = zeros(ik.iter_stop, 1);
        diff_x_tilde_op = zeros(ik.iter_stop, 1);
        diff_x_tilde_ellone = zeros(ik.iter_stop, 1);
        for i = 1:ik.iter_stop
            x_tilde_now = ik.x_tilde_est(:,:,i);
            diff_mat = x_tilde_now-oracle;

            diff_x_tilde_op(i)     = norm(diff_mat, 2);
            diff_x_tilde_ellone(i) = norm(diff_mat, 1);
            diff_x_tilde_fro(i)    = norm(diff_mat, "fro");
        end
    end
    %

  %  methods (Access = protected)
  %      function objective_value_original = get_objective_value_original(ik, cluster_est)
   %     objective_value_original = 0;
  %      for i = 1:ik.data_object.number_cluster
  %          cluster_size = sum(cluster_est==i);
 %          affinity_cluster = ik.data_object.sparse_affinity(cluster_est==i, cluster_est==i);
 %           within_cluster_variation = ((-2*sum(affinity_cluster, "all") + 2*cluster_size*trace(affinity_cluster))/cluster_size;
 %           objective_value_original = objective_value_original + within_cluster_variation;
 %       end

   % end
    %end

end %end of methods
end %end of classdef

    

