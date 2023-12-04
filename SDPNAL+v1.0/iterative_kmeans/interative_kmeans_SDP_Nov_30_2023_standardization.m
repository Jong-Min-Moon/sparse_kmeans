function [cluster_acc, abs_diff_normalized, thres] = interative_kmeans_SDP_Nov_30_2023_standardization(x, sigma, K, p, n_iter, rounding, n, cluster_true)     
%data generation
abs_diff_normalized = zeros([n_iter, p]);
thres = zeros(n_iter);
x_now = x;
A_now = (x_now * x_now')/ n;
Z_now = kmeans_sdp(A_now, K);     
cluster_est_now = estimate_cluster(Z_now, rounding, n, cluster_true);
cluster_acc_before_thres = max( mean(cluster_true ==  cluster_est_now), mean(cluster_true == -cluster_est_now));
fprintf("\np = %i, acc before thres: %f\n", p, cluster_acc_before_thres);
    
% iterate
for iter = 1:n_iter
    fprintf("%i th thresholding\n", iter)
    % 1. estimate cluster means
    n_g1_now = sum(cluster_est_now == 1);
    n_g2_now = sum(cluster_est_now ==-1);
    fprintf("n_G1=%i, n_G2 =%i\n", n_g1_now, n_g2_now )

    if max(n_g1_now, n_g2_now) == n
        fprintf("all observations are clustered into one group\n")
        cluster_acc = 0.5;
        % just report some values
        c = sqrt(2 * log(p) ) * sqrt(sigma^2*(1/max(1,n_g1_now) + 1/max(1,n_g2_now)));
        fprintf("threshold = %f\n", c)
        return 
    end

            
    x_now_g1 = x((cluster_est_now ==  1), :); 
    x_now_g2 = x((cluster_est_now == -1), :);
        
    x_bar_g1 = mean(x_now_g1, 1);  
    x_bar_g2 = mean(x_now_g2, 1);
            
    % 2. threshold the data matrix
    n_factor = sqrt( (n_g1_now*n_g2_now)/n );
    thres(iter) = sqrt(2 * log(p) );
    fprintf("threshold = %f\n", thres(iter))

    fprintf("normalized difference:")
    abs_diff_normalized(iter,:) = abs(x_bar_g1 - x_bar_g2) / sigma * n_factor;
    abs_diff_normalized(iter,:)
    fprintf("\n")

    
    thresholder_vec = abs_diff_normalized(iter,:) > thres(iter);
    fprintf("%i entries survived \n",sum(thresholder_vec))
    find(thresholder_vec)
           
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
    fprintf("\nAfter %ith thresholding, acc= %f\n", iter, cluster_acc_now);
    % end one iteration
end % end of iterative algorithm
cluster_acc = cluster_acc_now;