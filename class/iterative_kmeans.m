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
    cluster_est
    iter_stop
end

methods
    function ik = iterative_kmeans(data, estimator_class, number_cluster, omega_sparsity, init_method)
        ik.data_object = estimator_class(data, omega_sparsity);
        ik.number_cluster = number_cluster;
        ik.omega_sparsity = omega_sparsity;
        ik.init_method = init_method;
    end
    
    function learn(ik, max_n_iter) 
        ik.initialize_saving_matrix(max_n_iter)
  
        %initialization
        initial_cluster_assign = ik.get_initial_cluster_assign();
        ik.cluster_est(1,:) = initial_cluster_assign;
 
        for iter = 1:max_n_iter
            fprintf("\n%i th thresholding\n\n", iter)
            
            %estimation and thresholding
            tic
            [data_innovated_small, data_innovated_big, sample_covariance_small] = ik.data_object.threshold(ik.cluster_est(iter,:), ik.omega_sparsity);

            ik.omega_est_time(iter) = toc;
            ik.entries_survived(iter,:) = ik.data_object.support;
            ik.x_tilde_est(:,:,iter) = data_innovated_big;
        

            % solve SDP
            tic
            [Z_now, obj_val] = kmeans_sdp( data_innovated_small' * sample_covariance_small * data_innovated_small/ ik.data_object.sample_size, ik.data_object.number_cluster);
            clutser_est_vec = sdp_to_cluster(Z_now, ik.data_object.number_cluster);
            ik.sdp_solve_time(iter) = toc;
            ik.obj_val_prim(iter) = obj_val(1);
            ik.obj_val_dual(iter) = obj_val(2);
            obj_val(2)
            obj_og_now = ik.get_objective_value_original(clutser_est_vec);
            ik.obj_val_original(iter) = obj_og_now;
            ik.cluster_est(iter+1, :) = clutser_est_vec;
            fprintf("\n%i entries survived \n",sum(ik.data_object.support))
            
            if ik.is_stop(iter)
                ik.iter_stop = iter;
                break
            if iter == max_n_iter
                ik.iter_stop = iter
            end

            end

        end % end one iteration
    end

    function stop = is_stop(ik, iter)
        if iter == 1
            stop = false;
        else
            relative_change_original = abs((ik.obj_val_original(iter) - ik.obj_val_original(iter-1))/ik.obj_val_original(iter-1));
            relative_change_sdp = abs((ik.obj_val_dual(iter) - ik.obj_val_dual(iter-1))/ik.obj_val_dual(iter-1));
            if (relative_change_original < 0.01) & (relative_change_sdp < 0.01)
                stop = true;
            else
                stop = false;
            end
        end
    end
    function initialize_saving_matrix(ik, max_n_iter)
        p = ik.data_object.dimension;
        n = ik.data_object.sample_size;
        ik.x_tilde_est      = zeros(p, n, max_n_iter);
        ik.omega_est_time   = zeros(max_n_iter, 1);
        ik.sdp_solve_time   = zeros(max_n_iter, 1);
        ik.entries_survived = zeros(max_n_iter, p);
        ik.obj_val_prim     = zeros(max_n_iter, 1);
        ik.obj_val_dual     = zeros(max_n_iter, 1);
        ik.obj_val_original = zeros(max_n_iter, 1);
        ik.cluster_est      = zeros(max_n_iter+1, n);
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
        end
    end
    
    function objective_value_original = get_objective_value_original(ik, cluster_est)
        objective_value_original = 0;
        for i = 1:ik.data_object.number_cluster
            cluster_size = sum(cluster_est==i);
            affinity_cluster = ik.data_object.sparse_affinity(cluster_est==i, cluster_est==i);
            objective_value_original = objective_value_original + sum(affinity_cluster, "all");
        end

    end

    function database_subtable = get_database_subtable(ik, rep, Delta, rho, s, cluster_true, Omega)
        current_time = get_current_time();
        acc_vec = ik.evaluate_accuracy(cluster_true);
        [discov_true_vec, discov_false_vec, survived_indices] = ik.evaluate_discovery(s);
        [diff_x_tilde_fro, diff_x_tilde_op, diff_x_tilde_ellone] = ik.evaluate_innovation_est(Omega);
        %fprintf( strcat( "acc =", join(repelem("%f ", length(acc_vec))), "\n"),  acc_vec );
        cluster_string_vec = ik.get_cluster_string_vec();


        database_subtable = table(...
            repelem(rep, ik.iter_stop+1)',...                      % 01 replication number
            (0:ik.iter_stop)',...                                  % 02 step iteration number
            repelem(Delta, ik.iter_stop+1)',...                    % 03 separation
            repelem(ik.data_object.dimension, ik.iter_stop+1)',... % 04 data dimension
            repelem(rho, ik.iter_stop+1)',...                      % 05 conditional correlation
            repelem(s, ik.iter_stop+1)',...                        % 06 sparsity
            acc_vec,...                                      % 07 accuracy
            [0; ik.obj_val_prim(1:ik.iter_stop)],...                         % 08 objective function value (relaxed, primal)
            [0; ik.obj_val_dual(1:ik.iter_stop)],...                         % 09 objective function value (relaxed, dual)
            [0; ik.obj_val_original(1:ik.iter_stop)],...                     % 10 objective function value (original)
            [0; discov_true_vec],...                         % 11 true discovery
            [0; discov_false_vec],...                        % 12 false discovery
            [0; diff_x_tilde_fro],...                        % 13 estimation error of the innovated data, in Frobenius norm
            [0; diff_x_tilde_op],...                         % 14 estimation error of the innovated data, in operator norm
            [0; diff_x_tilde_ellone],...                     % 15 estimation error of the innovated data, in \ell_1 norm
            [0; ik.omega_est_time(1:ik.iter_stop)],...                       % 16 timing for estimating the precision matrix
            [0; ik.sdp_solve_time(1:ik.iter_stop)], ...                      % 17 timing elapsed for solving the SDP
            repelem(current_time, ik.iter_stop+1)', ...            % 18 timestamp
            [""; survived_indices],...                       % 19 indices of survived entry
            cluster_string_vec,...                           % 20 clustering information
            'VariableNames', ...
            ...  1      2       3      4      5        6         
            ["rep", "iter", "sep", "dim", "rho", "sparsity", ...
            ...  7        8           9             10             11               12
             "acc", "obj_prim", "obj_dual", "obj_original", "discov_true", "discov_false", ...
            ...       13              14                    15
             "diff_x_tilde_fro", "diff_x_tilde_op", "diff_x_tilde_ellone", ...
            ...  16          17           18
             "time_est", "time_SDP", "jobdate", ...
            ...      19              20            
            "survived_indices", "cluster_est"
            ]);
    end


    function acc_vec = evaluate_accuracy(ik, cluster_true)
        acc_vec = zeros(ik.iter_stop+1, 1);
        permutation_all = perms(1:ik.number_cluster);
        number_permutation = size(permutation_all, 1);
        for i = 1:(ik.iter_stop+1)
            cluster_est_now = ik.cluster_est(i,:);
            
            acc_vec(i) = max( mean(cluster_true == cluster_est_now), mean(cluster_true == (-cluster_est_now + 3)));
        end
    end

    function [discov_true_vec, discov_false_vec, survived_indices] = evaluate_discovery(ik, s)
        discov_true_vec = zeros(ik.iter_stop, 1);
        discov_false_vec = zeros(ik.iter_stop, 1);
        survived_indices = strings(ik.iter_stop, 1);
        for i = 1:ik.iter_stop
            entries_survived_now = ik.entries_survived(i,:);
            discov_true_vec(i) = sum(entries_survived_now(1:s));
            discov_false_vec(i) = sum(entries_survived_now) - discov_true_vec(i);
            survived_indices(i) = get_num2str_with_mark( find(entries_survived_now), ',');
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

    function cluster_string_vec = get_cluster_string_vec(ik)
        
        cluster_string_vec = strings(ik.iter_stop+1, 1);
        for i = 1:(ik.iter_stop+1)
            cluster_string_vec(i) = get_num2str_with_mark(ik.cluster_est(i,:), ',');
        end
    end

end %end of methods
end %end of classdef

    

