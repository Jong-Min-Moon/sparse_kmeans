function cluster_acc = iterative_kmeans(x, sigma, K, p, n_iter, rounding, n, cluster_true)     
%data generation
A = (x * x')/ n; % original affinity matrix (or similarity matrix)
                    %scaling
x_now = x;
A_now = (x_now * x_now')/ n;
Z_now = kmeans_sdp(A_now, K);     
cluster_est_now = estimate_cluster(Z_now, rounding, n, cluster_true);
cluster_acc_before_thres = max( ...
            mean(cluster_true ==  cluster_est_now), ...
            mean(cluster_true == -cluster_est_now) ...
            );
%fprintf("p = %i, acc before thres: %f", p, cluster_acc_before_thres);
    
% iterate
for iter = 1:n_iter
    %fprintf("%i th thresholding", iter)
    % 1. estimate cluster means
    n_g1_now = sum(cluster_est_now == 1);
    n_g2_now = sum(cluster_est_now ==-1);
            
    x_now_g1 = x((cluster_est_now ==  1), :); 
    x_now_g2 = x((cluster_est_now == -1), :);
        
    x_bar_g1 = mean(x_now_g1, 1);  
    x_bar_g2 = mean(x_now_g2, 1);
            
    % 2. threshold the data matrix
    c = sqrt(2 * log(p) ) * sqrt(sigma^2*(1/n_g1_now + 1/n_g2_now));
    thresholder_vec = abs(x_bar_g1 - x_bar_g2) > c;
    %fprintf("%i entries survived \n",sum(thresholder_vec))
    %find(thresholder_vec)
           
    x_tilde_g1 =  x_now_g1 * diag(thresholder_vec);
    x_tilde_g2 =  x_now_g2 * diag(thresholder_vec);
        
    % 3. apply SDP k-means
    x_now = zeros([n,p]);
    x_now((cluster_est_now == 1), :)  = x_tilde_g1;
    x_now((cluster_est_now == -1), :)  = x_tilde_g2;
    A_now = (x_now * x_now')/ n;
    Z_now = kmeans_sdp(A_now, K);     
    cluster_est_now = estimate_cluster(Z_now, rounding, n, cluster_true);
        
    cluster_acc_now = max( ...
                    mean(cluster_true == cluster_est_now), ...
                    mean(cluster_true == -cluster_est_now) ...
                    );
    % end one iteration
end % end of iterative algorithm
cluster_acc = cluster_acc_now;