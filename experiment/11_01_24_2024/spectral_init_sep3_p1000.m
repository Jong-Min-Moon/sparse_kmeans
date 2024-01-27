n=200;
sigma = 1;
K=2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
n_iter = 10; 

sep =3;
s = 10;
M = sqrt(sep^2 / 4 / s);
n_rep = 100;
p=1000;

clustering_acc_mat = repelem(0, n_rep );
tic
p=5000;
sparse_mean = [repelem(1,s), repelem(0,p-s)];
mu_1 =  -M * sparse_mean;
mu_2 =   M * sparse_mean;
   % norm(mu_1 - mu_2)
mu_1_mat = repmat(mu_1, n/2,1);
mu_2_mat = repmat(mu_2, n/2,1);
x_noiseless = [ mu_1_mat ; mu_2_mat ];
% parallel for loop


for j = 1:n_rep
    %data generation
    fprintf("%ith repeat")
    x_noisy = x_noiseless+ sigma^2 * randn(n, p);
    clustering_acc_mat(j) = iterative_kmeans_spectral_init_ver_01_18_24(x_noisy, sigma, K, 10, rounding, cluster_true);
   
        % iterate
 
        
end
toc


csvwrite('spectral_init_sep3_p_1000.csv',clustering_acc_mat)