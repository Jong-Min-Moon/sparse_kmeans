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
    cluster_est
end

methods
    function ik = iterative_kmeans(data, estimator_class, number_cluster, omega_sparsity, init_method)
        ik.data_object = estimator_class(data, omega_sparsity);
        ik.number_cluster = number_cluster;
        ik.omega_sparsity = omega_sparsity;
        ik.init_method = init_method;
    end
    
    function learn(ik, n_iter) 
        ik.initialize_saving_matrix(n_iter)
  
        %initialization
        initial_cluster_assign = ik.get_initial_cluster_assign();
        ik.cluster_est(1,:) = initial_cluster_assign;
 
        for iter = 1:n_iter
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
            ik.cluster_est(iter+1, :) = sdp_to_cluster(Z_now, ik.data_object.number_cluster);
            ik.sdp_solve_time(iter) = toc;
            ik.obj_val_prim(iter) = obj_val(1);
            ik.obj_val_dual(iter) = obj_val(2);

            fprintf("\n%i entries survived \n",sum(ik.data_object.support))
        
        end % end one iteration
    end

    function initialize_saving_matrix(ik, n_iter)
        p = ik.data_object.dimension;
        n = ik.data_object.sample_size;
        ik.x_tilde_est      = zeros(p, n, n_iter);
        ik.omega_est_time   = zeros(n_iter, 1);
        ik.sdp_solve_time   = zeros(n_iter, 1);
        ik.entries_survived = zeros(n_iter, p);
        ik.obj_val_prim     = zeros(n_iter, 1);
        ik.obj_val_dual     = zeros(n_iter, 1);
        ik.cluster_est      = zeros(n_iter+1, n);
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

    function acc_vec = evaluate_accuracy(ik, cluster_true)
        n_iter = size(ik.obj_val_prim,1);
        acc_vec = zeros(n_iter+1, 1);
        for i = 1:n_iter+1
            cluster_est_now = ik.cluster_est(i,:);
            acc_vec(i) = max( mean(cluster_true == cluster_est_now), mean(cluster_true == (-cluster_est_now + 3)));
        end
    end

    function [discov_true_vec, discov_false_vec] = evaluate_discovery(ik, s)
        n_iter = size(ik.obj_val_prim,1);
        discov_true_vec = zeros(n_iter, 1);
        discov_false_vec = zeros(n_iter, 1);
        for i = 1:n_iter
            entries_survived_now = ik.entries_survived(i,:);
            discov_true_vec(i) = sum(entries_survived_now(1:s));
            discov_false_vec(i) = sum(entries_survived_now) - discov_true_vec(i);
        end
    end

    function [diff_x_tilde_fro, diff_x_tilde_op, diff_x_tilde_ellone] = evaluate_innovation_est(ik,Omega)
        oracle = Omega * ik.data_object.data;
        n_iter = size(ik.obj_val_prim,1);
        diff_x_tilde_fro = zeros(n_iter, 1);
        diff_x_tilde_op = zeros(n_iter, 1);
        diff_x_tilde_ellone = zeros(n_iter, 1);
        for i = 1:n_iter
            x_tilde_now = ik.x_tilde_est(:,:,i);
            diff_mat = x_tilde_now-oracle;

            diff_x_tilde_op(i)     = norm(diff_mat, 2);
            diff_x_tilde_ellone(i) = norm(diff_mat, 1);
            diff_x_tilde_fro(i)    = norm(diff_mat, "fro");
        end
    end

end %end of methods
end %end of classdef

    

