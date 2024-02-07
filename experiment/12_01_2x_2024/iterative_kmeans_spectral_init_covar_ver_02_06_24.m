function cluster_acc = iterative_kmeans_spectral_init_covar_ver_01_26_24(x, Sigma, K, n_iter, cluster_true, init_method, verbose, sdp_method) 
% Sigma = known covariance matrix
%data generation
% created 01/26/2024
% modified 02/06/2024
init_method
sdp_method
% spectral initialization
n = size(x,2);
p = size(x,1);
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
end
cluster_est_now = cluster_est_now .* (cluster_est_now ~= 2) + (cluster_est_now == 2)* (-1);
cluster_est_now = cluster_est_now';

thres = sqrt(2 * log(p) );
Omega = inv(Sigma);
Omega_diag = diag(Omega);
X_tilde =   Omega  *  x;
X_tilde_now =     X_tilde;

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


    if max(n_g1_now, n_g2_now) ==n
        %fprintf("all observations are clustered into one group")
        cluster_acc = 0.5;
        return 
    end
    
    
            
    X_tilde_now_g1 = X_tilde(:, (cluster_est_now ==  1)); 
    X_tilde_now_g2 = X_tilde(:, (cluster_est_now == -1));
        
    X_tilde_bar_g1 = mean(X_tilde_now_g1, 2);  
    X_tilde_bar_g2 = mean(X_tilde_now_g2, 2);
            
    % 2. threshold the data matrix
    abs_diff = abs(X_tilde_bar_g1 - X_tilde_bar_g2)./sqrt(Omega_diag) * sqrt( n_g1_now*n_g2_now/n );
    abs_diff_sort = -sort(-abs_diff);
    top_10 = abs_diff_sort(1:10);

    
    s_hat = abs_diff > thres;
    n_entries_survived = sum(s_hat);

    if n_entries_survived == 0
        cluster_acc = 0.5;
        break
    end
    
    Sigma_s_hat_now = Sigma(s_hat,s_hat);
    X_tilde_now  = X_tilde(s_hat,:);  

  
    Z_now = kmeans_sdp( X_tilde_now' * Sigma_s_hat_now * X_tilde_now/ n, K);       

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
        fprintf("%i entries survived \n\n",n_entries_survived)
        fprintf("normalized difference top 10 max: (%f) * sigma \n", top_10)
        fprintf("n_{G1}_now = %i, n_{G1}_now = %i\n", n_g1_now, n_g2_now )
        fprintf("acc_now= %f", cluster_acc_now);
        disp(find(s_hat))
    end
    % end one iteration
end % end of iterative algorithm
cluster_acc = cluster_acc_now