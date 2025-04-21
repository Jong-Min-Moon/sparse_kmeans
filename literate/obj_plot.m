sep=2;
n=200;
p=1000;
s_list = [6, 32, 210];
cluster_1_ratio_list = [0.1, 0.3, 0.5];
n_draw = 2000;
baseline = 2;
for s = s_list
    s

    for cluster_1_ratio = cluster_1_ratio_list 
        beta_seed = randi(999999)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %half overlap, same size     
        1
        compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((ceil(s/2) + 1):(s + ceil(s/2))), n_draw, beta_seed);
        
        %half overlap, many noise  
        2
        compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((ceil(s/2) + 1):(s + ceil(s/2) + ceil(s/2) )) , n_draw, beta_seed);
        
        % half overlap, small noise
        3
        compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ( (ceil(s/2) + 1): (s + ceil(s/2)- floor(3*s/4)) ) , n_draw, beta_seed);
        
        % pure noise, same size
        4
        compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((s+1): (2*s)), n_draw, beta_seed);
        
        % pure noise, larger size
        5
        compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((s+1): (2*s + ceil(s/2))), n_draw, beta_seed);
        
        % pure noise, small size
        6
        compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, 1:s, ((s+1):(2*s - floor(3*s/4))), n_draw, beta_seed);
    end
end