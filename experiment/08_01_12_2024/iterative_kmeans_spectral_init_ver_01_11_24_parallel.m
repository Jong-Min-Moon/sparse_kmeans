function cluster_acc = iterative_kmeans_ESSC_init_ver_01_11_24_parallel(x, sigma, K, n_iter, rounding, cluster_true)     
%data generation
% created 01/11/2024
n = size(x,1);
p = size(x,2);
H_hat = (x * x');
[V,D] = eig(H_hat);
[d,ind] = sort(diag(D));
Ds = D(ind,ind);
Vs = V(:,ind);
[idx,C] = kmeans(Vs(:,1:10),K);
cluster_est_now = idx .* (idx ~= 2) + (idx == 2)* (-1);
cluster_est_now = cluster_est_now'
thres = sqrt(2 * log(p) );

cluster_acc_before_thres = max( mean(cluster_true ==  cluster_est_now), mean(cluster_true == -cluster_est_now));
%fprintf("\np = %i, acc_init: %f \n", p, cluster_acc_before_thres);
n_g1_now = sum(cluster_est_now == 1);
n_g2_now = sum(cluster_est_now ==-1); 
%fprintf("n_{G1}_init = %i, n_{G1}_init = %i\n", n_g1_now, n_g2_now )
%fprintf("threshold: (%f)\n", thres)
% iterate
for iter = 1:n_iter
    %fprintf("\n%i th thresholding\n\n", iter)
    % 1. estimate cluster means


    if max(n_g1_now, n_g2_now) ==n
     %   fprintf("all observations are clustered into one group")
        cluster_acc = 0.5;
        return 
    end

            
    x_now_g1 = x((cluster_est_now ==  1), :); 
    x_now_g2 = x((cluster_est_now == -1), :);
        
    x_bar_g1 = mean(x_now_g1, 1);  
    x_bar_g2 = mean(x_now_g2, 1);
            
    % 2. threshold the data matrix

       
    abs_diff = abs(x_bar_g1 - x_bar_g2) * sqrt( n_g1_now*n_g2_now/n ) / sigma;
    %abs_diff_sort = -sort(-abs_diff);
    %top_10 = abs_diff_sort(1:10);
    %fprintf("normalized difference top 10 max: (%f) * sigma \n", top_10)

    thresholder_vec = abs_diff > thres;
    %fprintf("%i entries survived \n\n",sum(thresholder_vec))
    %find(thresholder_vec)
        
    % 3. apply SDP k-means
    x_now = zeros([n,p]);
    x_now((cluster_est_now == 1), :)  = x_now_g1 * diag(thresholder_vec);
    x_now((cluster_est_now == -1), :)  = x_now_g2 * diag(thresholder_vec);
    Z_now = kmeans_sdp( (x_now * x_now')/ n, K);     
    cluster_est_now = estimate_cluster(Z_now, rounding, n, cluster_true);
        
    cluster_acc_now = max( ...
                    mean(cluster_true == cluster_est_now), ...
                    mean(cluster_true == -cluster_est_now) ...
                    );
    n_g1_now = sum(cluster_est_now == 1);
    n_g2_now = sum(cluster_est_now ==-1);
    %fprintf("n_{G1}_now = %i, n_{G1}_now = %i\n", n_g1_now, n_g2_now )
    %fprintf("acc_now= %f", cluster_acc_now);
    % end one iteration
end % end of iterative algorithm
cluster_acc = cluster_acc_now;