function [cluster_est, diff_x_tilde, diff_omega_diag, entries_survived, omega_est_time, sdp_solve_time, obj_val] = iterative_kmeans_ISEE_denoise(x, K, n_iter, Omega, omega_sparsity, init_method) 

    % modified 03/07/2024
    n     = size(x,2);
    p     = size(x,1);
    lambda = sqrt(log(p)/n);
    diverging_quantity = sqrt(log(n));
    thres = diverging_quantity*max(omega_sparsity*lambda^2, lambda);

    fprintf("thres=%f", thres)
    Omega_x = Omega * x;
    %saving arrays
    diff_x_tilde     = zeros(n_iter, 1);
    diff_omega_diag  = zeros(n_iter, 1);
    omega_est_time   = zeros(n_iter, 1);
    sdp_solve_time   = zeros(n_iter, 1);
    entries_survived = zeros(n_iter, p);
    obj_val          = zeros(n_iter, p);
    cluster_est      = zeros(n_iter+1, n);
    
    %initialization
    cluster_est_now = initialize_bicluster(x, init_method);
    cluster_est(1,:) = cluster_est_now;

    for iter = 1:n_iter
        fprintf("\n%i th thresholding\n\n", iter)
        % 1. estimate cluster means

        % complete failure
        n_g1_now = sum(cluster_est_now == 1);
        n_g2_now = sum(cluster_est_now ==-1);
        if max(n_g1_now, n_g2_now) == n
            fprintf("all observations are clustered into one group")
            return 
        end

        % innovated data estimation
        tic
        [mean_now, noise_now, Omega_diag_hat] = ISEE_bicluster(x, cluster_est_now);
        omega_est_time(iter) = toc
        x_tilde_now = mean_now + noise_now;
        diff_x_tilde(iter) = norm(x_tilde_now-Omega_x, "fro")
        diff_omega_diag(iter) = norm(Omega_diag_hat-diag(Omega), "fro")
        
        % 2. threshold the data matrix
        signal_est_now = mean( mean_now(:, cluster_est_now==1), 2) - mean( mean_now(:, cluster_est_now==-1), 2);   
        abs_diff = abs(signal_est_now);
        s_hat = abs_diff > thres;
        x_tilde_now_s  = x_tilde_now(s_hat,:);
        entries_survived(iter,:) = s_hat;
        n_entries_survived = sum(s_hat);


        % complete failure
        if n_entries_survived == 0
            disp("no entry survived")
            break
        end

        % sigma hat s estimation
        X_g1_now = x(:, (cluster_est_now ==  1)); 
        X_g2_now = x(:, (cluster_est_now ==  -1)); 
        X_mean_g1_now = mean(X_g1_now, 2);
        X_mean_g2_now = mean(X_g2_now, 2);
        data_full = [(X_g1_now - X_mean_g1_now) (X_g2_now - X_mean_g2_now)]';
        data_filtered = data_full(:,s_hat);
        Sigma_hat_s_hat_now = data_filtered' * data_filtered/(n-1);

        % solve SDP
        tic
        Z_now = kmeans_sdp( x_tilde_now_s' * Sigma_hat_s_hat_now * x_tilde_now_s/ n, K);
        cluster_est_now = sdp_to_cluster(Z_now, K);
        cluster_est(iter+1, :) = cluster_est_now;
        sdp_solve_time(iter) = toc
        

        fprintf("\n%i entries survived \n",n_entries_survived)
        
    end % end one iteration
    diff_x_tilde
    diff_omega_diag
    omega_est_time
    sdp_solve_time
