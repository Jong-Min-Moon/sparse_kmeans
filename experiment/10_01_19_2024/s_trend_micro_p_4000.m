n=200;
sigma = 1;
K=2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
n_iter = 15; 

p = 4000;

sep =3.5;
s_init = 10;
s_vec = 10:25;
M = sqrt(sep^2 / 4 / s_init);
n_rep = 100;


clustering_acc_mat = zeros(n_rep, length(s_vec));


for i = 1:length(s_vec)
    s_now = s_vec(i);
    fprintf("sparsity = (%i) \n", s_now)
    sparse_mean = [repelem(1,s_now), repelem(0,p-s_now)];
    mu_1 =  -M * sparse_mean;
    mu_2 =   M * sparse_mean;

    tic
    x_noiseless = [ repmat(mu_1, n/2,1) ; repmat(mu_2, n/2,1)];
    for j = 1:n_rep
        fprintf("iteration: (%i)th \n\n", j)
    %data generation
        x_noisy = x_noiseless+ sigma^2 * randn(n, p);
        clustering_acc_mat(j,i) = iterative_kmeans_spectral_init_ver_01_18_24(x_noisy, sigma, K, n_iter, rounding, cluster_true);
    toc
        % iterate
    end
end
csvwrite('spectral_init_sep35_p_4000_s_trend_micro.csv',clustering_acc_mat)