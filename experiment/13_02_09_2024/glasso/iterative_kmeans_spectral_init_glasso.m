function cluster_acc = iterative_kmeans_spectral_init_glasso(x, K, s, n_iter, cluster_true, init_method, verbose, sdp_method) 
% Sigma = UNknown covariance matrix
%data generation
% created 01/26/2024
init_method
sdp_method

% spectral initialization
n = size(x,2);
p = size(x,1);
thres = sqrt(2 * log(p) );

if strcmp(init_method, 'spec')
    H_hat = (x' * x)/n;
    [V,D] = eig(H_hat);
    [d,ind] = sort(diag(D), "descend");
    Ds = D(ind,ind);
    Vs = V(:,ind);
    [cluster_est_now,C] = kmeans(Vs(:,1),K);
elseif strcmp(init_method, "hc")
    Z = linkage(x', 'ward');
    cluster_est_now = cluster(Z, 'Maxclust',K);
elseif strcmp(init_method, "SDP")
    Sigma_est_now = cov(x');
    X_tilde_now = linsolve(Sigma_est_now, x);
    Z_now = kmeans_sdp( x'* X_tilde_now/ n, K);       
    
    % final thresholding
    [U_sdp,~,~] = svd(Z_now);
    U_top_k = U_sdp(:,1:K);
    [cluster_est_now,C] = kmeans(U_top_k,K);  % label

end
cluster_est_now = cluster_est_now .* (cluster_est_now ~= 2) + (cluster_est_now == 2)* (-1);
cluster_est_now = cluster_est_now';

cluster_acc_before_thres = max( mean(cluster_true ==  cluster_est_now), mean(cluster_true == -cluster_est_now));



n_g1_now = sum(cluster_est_now == 1);
n_g2_now = sum(cluster_est_now ==-1);

if verbose
    fprintf("\np = %i, acc_init: %f \n", p, cluster_acc_before_thres);
    fprintf("n_{G1}_init = %i, n_{G1}_init = %i\n", n_g1_now, n_g2_now )
    fprintf("threshold: (%f)\n", thres)
    
end
cluster_acc_now = cluster_acc_before_thres;
for iter = 1:n_iter
    if verbose
        fprintf("\n%i th thresholding\n\n", iter)
    end
    % 1. estimate cluster means


    if max(n_g1_now, n_g2_now) == n
        %fprintf("all observations are clustered into one group")
        cluster_acc = 0.5;
        return 
    end
    
    % covariance estimation
    X_g1_now = x(:, (cluster_est_now ==  1)); 
    X_g2_now = x(:, (cluster_est_now ==  -1));
    
    X_mean_g1_now = mean(X_g1_now, 2);
    X_mean_g2_now = mean(X_g2_now, 2);
    sample_cov =  [(X_g1_now - X_mean_g1_now) (X_g2_now - X_mean_g2_now)] * [(X_g1_now - X_mean_g1_now) (X_g2_now - X_mean_g2_now)]' / (n-1);
    
    lambda_vec = [0.2, 0.4, 0.6, 0.8];
    bic_vec = [0,0,0,0];
    for i = 1:4
    	lambda = lambda_vec(i);
    	[Omega_est_now, Sigma_est_glasso_now] = graphicalLasso(sample_cov, lambda);
    	bic_vec(i) = bic(Sigma_est_glasso_now, Omega_est_now, n, p);
    end
    [lambda_sort, lambda_sort_index]= sort(bic_vec)
    [Omega_est_now, Sigma_est_now] = graphicalLasso(sample_cov, lambda_sort(4));
    %heatmap(Omega_est_now)
    Omega_est_now_diag = diag(Omega_est_now)/n/3;
    Omega_est_now_diag(1)
  
            
    % 2. threshold the data matrix
    signal_est_now = Omega_est_now* (X_mean_g1_now - X_mean_g2_now);
    abs_diff = abs(signal_est_now)./sqrt(Omega_est_now_diag) * sqrt( n_g1_now*n_g2_now/n );
    [abs_diff_sort, abs_diff_sort_idx]= sort(abs_diff, "descend");
    top_10 =  abs_diff_sort(1:10)
    top_10_idx = abs_diff_sort_idx(1:10)

    
    s_hat = abs_diff > thres;
    entries_survived = find(s_hat);
    n_entries_survived = sum(s_hat);

    if n_entries_survived == 0
        disp("no entry survived")
        cluster_acc = 0.5;
        break
    end
    
    Sigma_hat_s_hat_now = Sigma_est_now(s_hat,s_hat);
    X_tilde_now = linsolve(Sigma_est_now, x);
    X_tilde_now  = X_tilde_now(s_hat,:);  


    Z_now = kmeans_sdp( X_tilde_now' * Sigma_hat_s_hat_now * X_tilde_now/ n, K);       
    
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
    if verbose
        fprintf("\n%i entries survived \n",n_entries_survived)
        fprintf("right : (%i)\n", sum(s_hat(1:s)))
        fprintf("wrong : (%i)\n", sum(s_hat(s+1:end)))
        fprintf("normalized difference top 10 max: (%f)\n", top_10)
        fprintf("normalized difference top 10 max index: (%i)\n", top_10_idx)
        fprintf("n_{G1}_now = %i, n_{G1}_now = %i\n", n_g1_now, n_g2_now )
        fprintf("acc_now= %f", cluster_acc_now);

    end
    % end one iteration
end % end of iterative algorithm
cluster_acc = cluster_acc_now
