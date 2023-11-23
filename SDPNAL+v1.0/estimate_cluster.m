function cluster_est = estimate_cluster(Z_org, rounding, n, cluster_true)        

rounded_cluster_one = Z_org>=rounding;
cluster_one_est = find(rounded_cluster_one(:,1));

rounded_cluster_two = Z_org<rounding;
cluster_two_est = find(rounded_cluster_two(:,1));
    
cluster_est = 1:n;
cluster_est(cluster_one_est) = 1;
cluster_est(cluster_two_est) = -1;

mean_og = mean(cluster_true == cluster_est);
mean_flipped = mean(cluster_true == -cluster_est);
if mean_og < mean_flipped
    cluster_est = -cluster_est
end
