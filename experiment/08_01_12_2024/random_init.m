function cluster_init_best = random_init(x, sigma, K, n_init)
n = size(x,1);
p = size(x,2);
thres = sqrt(2 * log(p) );
pd = makedist('Multinomial', 'Probabilities', repelem(1/K, K))
cluster_init = pd.random(n_init, n);
cluster_init = cluster_init .* (cluster_init ~= 2) + (cluster_init == 2)* (-1);
n_survive_vec = repelem(0,n_init);
for init_num = 1:n_init
    cluster_init_now = cluster_init(init_num,:);
    x_now_g1 = x((cluster_init_now ==  1), :); 
    x_now_g2 = x((cluster_init_now == -1), :);
        
    x_bar_g1 = mean(x_now_g1, 1);  
    x_bar_g2 = mean(x_now_g2, 1);
    n_g1_now = sum(cluster_init_now == 1);
    n_g2_now = sum(cluster_init_now ==-1);            
       
    abs_diff = abs(x_bar_g1 - x_bar_g2) * sqrt( n_g1_now*n_g2_now/n ) / sigma;
    n_survive_vec(init_num) = sum(abs_diff > thres);
end

[survive_max, survive_max_idx] = max(n_survive_vec)
cluster_init_best = cluster_init(survive_max_idx,:);
