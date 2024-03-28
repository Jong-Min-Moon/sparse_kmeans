function [cluster_est, diff_x_tilde, diff_omega_diag, omega_est_time, sdp_solve_time, obj_val_prim, obj_val_dual] = non_iter_kmeans(x, K, Omega, init_method) 

    % modified 03/022/2024
    n     = size(x,2);
    p     = size(x,1);
    

    Omega_x = Omega * x;
    
    %initialization: necessary of covariance estimation
    cluster_est_now = initialize_bicluster(x, init_method);


    % 1. estimate cluster means
    n_g1_now = sum(cluster_est_now == 1);
    n_g2_now = sum(cluster_est_now ==-1);
    % complete failure
    if max(n_g1_now, n_g2_now) == n
        fprintf("all observations are clustered into one group")
        return 
    end

    % innovated data estimation
    tic
    [mean_now, noise_now, Omega_diag_hat] = ISEE_bicluster(x, cluster_est_now);
    omega_est_time = toc;
    x_tilde_now = mean_now + noise_now;
    diff_x_tilde = norm(x_tilde_now-Omega_x, "fro");
    diff_omega_diag = norm(Omega_diag_hat-diag(Omega), "fro");
        



    X_g1_now = x(:, (cluster_est_now ==  1)); 
    X_g2_now = x(:, (cluster_est_now ==  -1)); 
    X_mean_g1_now = mean(X_g1_now, 2);
    X_mean_g2_now = mean(X_g2_now, 2);
    data_full = [(X_g1_now - X_mean_g1_now) (X_g2_now - X_mean_g2_now)]';  
    Sigma_hat_now = data_full' * data_full/(n-1);

    % solve SDP
    tic
    [Z_now, obj_val] = kmeans_sdp( x_tilde_now' * Sigma_hat_now * x_tilde_now/ n, K);
    cluster_est = sdp_to_cluster(Z_now, K);
    sdp_solve_time = toc;
    obj_val_prim = obj_val(1);
    obj_val_dual = obj_val(2);

        

