p=4000;
n=200;
sigma = 1;
K=2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
n_iter = 10; 

sep =3.5;
s_vec = [10,20,30,40,50];
s_init = 10;
M = 0.5*sep/sqrt(s_init);
n_rep = 100;
clustering_acc_mat_4000 = zeros(n_rep, length(s_vec));


for i = 1:length(s_vec)
    s_now = s_vec(i);
    fprintf("sparsity = %i \n", s_now)
    sparse_mean = [repelem(1,s_now), repelem(0,p-s_now)];
    
    mu_1 =  -M * sparse_mean;
    mu_2 =   M * sparse_mean;
    x_noiseless = [ repmat(mu_1, n/2,1); repmat(mu_2, n/2,1) ];
    tic
    for j = 1:n_rep
        rng(j)
        fprintf("iteration: (%i)th \n\n", j)
        x_noisy = x_noiseless+ sigma^2 * randn(n, p); %data generation
        clustering_acc_mat_4000(j,i) = iterative_kmeans_spectral_init_ver_01_19_24(x_noisy, sigma, K, n_iter, rounding, cluster_true);
    end
    toc
end

csvwrite('spectral_init_p_4000_s_trend_1.csv',clustering_acc_mat_4000)