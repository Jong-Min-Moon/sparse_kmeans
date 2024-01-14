n=200;
sigma = 1;
K=2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
n_iter = 10; 
%%%%%%%%%%%%
sep = 5;
%%%%%%%%%%%%%
s = 10;
M = sqrt(sep^2 / 4 / s);

p=3000
sparse_mean = [repelem(1,s), repelem(0,p-s)];
mu_1 =  -M * sparse_mean;
mu_2 =   M * sparse_mean;
mu_1_mat = repmat(mu_1, n/2,1);
mu_2_mat = repmat(mu_2, n/2,1);
    
        
x_noiseless = [ mu_1_mat ; mu_2_mat ];

e=eig(x_noiseless*x_noiseless');
e_sorted = -sort(-e)



x_noisy = [ mu_1_mat ; mu_2_mat ] + sigma^2 * randn(n, p);
e_noisy=eig(x_noisy*x_noisy');
e_noisy_sorted = -sort(-e_noisy)
hist(e_noisy)