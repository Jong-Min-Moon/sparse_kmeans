addpath(genpath('/mnt/nas/users/user213/sparse_kmeans'))
feature("numcores")
maxNumCompThreads(2);
% standard code for del3_rho05


p=9000;
Delta=3;
s = 10;
n_rep = 100;


n=200;
K=2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
n_iter = 10; 



clustering_acc_mat = repelem(0, n_rep );






Sigma = zeros([p,p]);
rho = 0.5;
for j=1:p
    for l = 1:p
        Sigma(j,l) = rho^(abs(j-l));
    end
end
Sigma_half = Sigma^(1/2);
Sigma_half_inv = inv(Sigma_half);


%M = sqrt(Delta^2 / 4 / s)
M = Delta/2/norm(sum(Sigma_half(:,1:s),2))
sparse_mean = [repelem(1,s), repelem(0,p-s)]'; %column vector
mu_0_tilde =  M * sparse_mean;
mu_0 = Sigma*mu_0_tilde;
mu_1 = -mu_0;
mu_2 = mu_0;
norm(Sigma_half_inv*(mu_1-mu_2))
norm((mu_1-mu_2))
tic

% norm(mu_1 - mu_2)


% parallel for loop

mu_1_mat = repmat(mu_1,  1, n/2); %each column is one observation
mu_2_mat = repmat(mu_2, 1, n/2);%each column is one observation
x_noiseless = [ mu_1_mat  mu_2_mat ];%each column is one observation 
x_noisy = x_noiseless +  mvnrnd(zeros(p,1), Sigma, n)';

for j = 1:n_rep
    fprintf("iteration: (%i)th \n\n", j)
    rng(j)
    tic
    %data generation
    
    x_noisy = x_noiseless +  mvnrnd(zeros(p,1), Sigma, n)';%each column is one observation
    clustering_acc_mat(j) = iterative_kmeans_spectral_init_covar_ver_01_26_24(x_noisy, Sigma, K, 10, cluster_true, 'spec', false, 'basic');
    acc_so_far =  clustering_acc_mat(1:j);
    fprintf( "mean acc so far: %f\n",  mean( acc_so_far ) );

    toc
        % iterate        
end
csvwrite('/mnt/nas/users/user213/sparse_kmeans/experiment/12_01_2x_2024/anisotropic_cov/del3/result/del3_rho05_p9000.csv',clustering_acc_mat)