function [cluster_est, diff_x_tilde, entries_survived, omega_est_time, sdp_solve_time, obj_val_prim, obj_val_dual] = iterative_kmeans_ISEE_clean(x, K, n_iter, Omega, omega_sparsity, init_method) 

    % modified 03/022/2024
    n     = size(x,2);
    p     = size(x,1);
    


    Omega_x = Omega * x;
    %saving arrays
    diff_x_tilde     = zeros(n_iter, 1);
    omega_est_time   = zeros(n_iter, 1);
    sdp_solve_time   = zeros(n_iter, 1);
    entries_survived = zeros(n_iter, p);
    obj_val_prim          = zeros(n_iter, 1);
    obj_val_dual          = zeros(n_iter, 1);
    cluster_est      = zeros(n_iter+1, n);
    
    %initialization
    cluster_est_now = initialize_bicluster(x, init_method);
    cluster_est(1,:) = cluster_est_now;
    data = data_gaussian_ISEE_clean(x,false);
    for iter = 1:n_iter
        fprintf("\n%i th thresholding\n\n", iter)
        % 1. estimate cluster means

        % complete failure


        tic
        [data_innovated_small, data_innovated_big, sample_covariance_small] = data.threshold(cluster_est_now, omega_sparsity);

        omega_est_time(iter) = toc;
        entries_survived(iter,:) = data.support;
        diff_x_tilde(iter) = norm(data_innovated_big-Omega_x, "fro");
        

        % solve SDP
        tic
        [Z_now, obj_val] = kmeans_sdp( data_innovated_small' * sample_covariance_small * data_innovated_small/ n, K);
        cluster_est_now = sdp_to_cluster(Z_now, K);
        cluster_est(iter+1, :) = cluster_est_now;
        sdp_solve_time(iter) = toc;
        obj_val_prim(iter) = obj_val(1);
        obj_val_dual(iter) = obj_val(2);

        fprintf("\n%i entries survived \n",sum(data.support))
        
    end % end one iteration

