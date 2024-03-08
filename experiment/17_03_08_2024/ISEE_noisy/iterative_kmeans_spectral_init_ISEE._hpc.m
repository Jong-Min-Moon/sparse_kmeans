function [cluster_acc, diff_x_tilde, diff_omega_diag, false_discov, true_discov, false_discov_top5, omega_est_time, sdp_solve_time] = iterative_kmeans_spectral_init_ISEE(x, K, n_iter, init_method, verbose) 

    import java.util.TimeZone

%data generation
% modified 03/07/2024
init_method


% spectral initialization
n = size(x,2);
p = size(x,1);
thres = sqrt(2 * log(p) )
fprintf("thres=%f", thres)

%saving arrays
diff_x_tilde    = zeros(n_iter, 1);
diff_omega_diag = zeros(n_iter, 1);
false_discov    = zeros(n_iter, 1);
true_discov     = zeros(n_iter, 1);
omega_est_time  = zeros(n_iter, 1);
sdp_solve_time  = zeros(n_iter, 1);
cluster_est    = zeros(n_iter, n);

cluster_est_now = initialize_bicluster(x, init_method);
cluster_est(1,:) = cluster_est_now
%cluster_acc_before_thres = max( mean(cluster_true ==  cluster_est_now), mean(cluster_true == -cluster_est_now));



n_g1_now = sum(cluster_est_now == 1);
n_g2_now = sum(cluster_est_now ==-1);

if verbose
    %fprintf("\np = %i, acc_init: %f \n", p, cluster_acc_before_thres);
    fprintf("n_{G1}_init = %i, n_{G1}_init = %i\n", n_g1_now, n_g2_now )
    fprintf("threshold: (%f)\n", thres)
    
end
%cluster_acc_now = cluster_acc_before_thres;

for iter = 1:n_iter
    if verbose
        fprintf("\n%i th thresholding\n\n", iter)
    end
    % 1. estimate cluster means


    if max(n_g1_now, n_g2_now) == n
        %fprintf("all observations are clustered into one group")
        %cluster_acc = 0.5;
        return 
    end
    tic
    [mean_now, noise_now, Omega_diag_hat] = ISEE_bicluster(x, cluster_est_now);
    omega_est_time(iter) = toc
    x_tilde_now = mean_now + noise_now;
    diff_x_tilde(iter) = norm(x_tilde_now-Omega_x, "fro")
    diff_omega_diag(iter) = norm(Omega_diag_hat-diag(Omega), "fro")
    % 2. threshold the data matrix

    %signal_est_now = mean( mean_now(:, cluster_est_now==1), 2) - mean( mean_now(:, cluster_est_now==-1), 2);
    %abs_diff = signal_est_now;
    %thres = 0.1;

    signal_est_now = mean( x_tilde_now(:, cluster_est_now==1), 2) - mean( x_tilde_now(:, cluster_est_now==-1), 2);   
    abs_diff = abs(signal_est_now)./sqrt(Omega_diag_hat) * sqrt( n_g1_now*n_g2_now/n );
    [abs_diff_sort, abs_diff_sort_idx]= sort(abs_diff, "descend");
    discov_idx_sorted = abs_diff_sort_idx(abs_diff_sort>thres);
    false_discov_idx_sorted = discov_idx_sorted(discov_idx_sorted>s);

    s_hat = abs_diff > thres;
    n_entries_survived = sum(s_hat);
        entries_survived = find(s_hat);


    if n_entries_survived == 0
        disp("no entry survived")
        cluster_acc = 0.5;
        break
    end

    %estimate sigma hat s
    X_g1_now = x(:, (cluster_est_now ==  1)); 
    X_g2_now = x(:, (cluster_est_now ==  -1)); 
    X_mean_g1_now = mean(X_g1_now, 2);
    X_mean_g2_now = mean(X_g2_now, 2);
    data_py = [(X_g1_now - X_mean_g1_now) (X_g2_now - X_mean_g2_now)]';
    data_filtered = data_py(:,s_hat);
    Sigma_hat_s_hat_now = data_filtered' * data_filtered/(n-1);


 
    x_tilde_now_s  = x_tilde_now(s_hat,:);  

    tic
    Z_now = kmeans_sdp( x_tilde_now_s' * Sigma_hat_s_hat_now * x_tilde_now_s/ n, K);
    sdp_solve_time(iter) = toc
    
    % final thresholding
    [U_sdp,~,~] = svd(Z_now);
    U_top_k = U_sdp(:,1:K);
    [cluster_est_now,C] = kmeans(U_top_k,K);  % label
    cluster_est_now = cluster_est_now .* (cluster_est_now ~= 2) + (cluster_est_now == 2)* (-1);    
    cluster_est_now = cluster_est_now';   
    cluster_acc_now = max( ...
                    mean(cluster_true == cluster_est_now), ...
                    mean(cluster_true == -cluster_est_now) ...
                    );
    n_g1_now = sum(cluster_est_now == 1);
    n_g2_now = sum(cluster_est_now ==-1);

    top_num = min(5, length(false_discov_idx_sorted));
    if top_num > 0
        false_discov_top5(iter) = strjoin(arrayfun(@(x) num2str(x), false_discov_idx_sorted(1:top_num) ,'UniformOutput',false),'_')
    end
    
        fprintf("right : (%i)\n", sum(s_hat(1:s)))
    fprintf("wrong : (%i)\n", sum(s_hat(s+1:end)))  
    false_discov(iter) = sum(s_hat(s+1:end));
    true_discov(iter) = sum(s_hat(1:s));
    
    if verbose
        fprintf("\n%i entries survived \n",n_entries_survived)

        %fprintf("normalized difference top 10 max: (%f)\n", top_10)
        %fprintf("normalized difference top 10 max index: (%i)\n", top_10_idx)
        fprintf("n_{G1}_now = %i, n_{G1}_now = %i\n", n_g1_now, n_g2_now )
        fprintf("acc_now= %f", cluster_acc_now);
    
    % into sql
    n = now;
    ds = datestr(n);
    dt = datetime(ds,'TimeZone',char(TimeZone.getDefault().getID()));

    data = table("aaa",  rep,   iter,    5, 500, 0.45, 10, 0.9854, 5, 10, 2.3, 3, 1, 1, dt, 'VariableNames', ...
	[            "job", "rep", "iter", "sep", "dim", "rho", "sparsity", "acc", "discov_true", "discov_false", "diff_x_tilde", "diff_omega_diag",  "time_isee", "time_SDP", "jobdate"])
    end
    % end one iteration
end % end of iterative algorithm
cluster_acc = cluster_acc_now
omega_est_time
diff_x_tilde
diff_omega_diag
false_discov
true_discov
false_discov_top5
sdp_solve_time
