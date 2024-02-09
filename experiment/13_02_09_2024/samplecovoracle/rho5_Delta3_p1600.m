rho = 5;
p = 1600
Delta = 3
path_result = '/mnt/nas/users/user213/sparse_kmeans/experiment/13_02_09_2024/samplecovoracle/result/rho5_Delta3_p1600.csv'

%p=
%Delta=
%rho = 
rho = rho /100
addpath(genpath('/mnt/nas/users/user213/sparse_kmeans'))
feature("numcores")
maxNumCompThreads(1);




s = 10;
n_rep = 100;


n=200;
K=2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
n_iter = 10; 



clustering_acc_mat = repelem(0, n_rep );






Omega = zeros([p,p]);

for j=1:p
    for l = 1:p
        if j==l
            Omega(j,l) = 1;
        elseif abs(j-l) ==1
            Omega(j,l) = rho;
        end
    end
end



try chol(Omega);
    disp('Matrix is symmetric positive definite.')
catch ME
    disp('Matrix is not symmetric positive definite')
end

Sigma = inv(Omega);
M = Delta/2/ sqrt( sum( Sigma(1:s,1:s),"all") )
sparse_mean = [repelem(1,s), repelem(0,p-s)]'; %column vector
mu_0_tilde =  M * sparse_mean;
mu_0 = Sigma*mu_0_tilde;
mu_1 = -mu_0;
mu_2 = mu_0;

beta = Omega * (mu_1-mu_2);
fprintf( "delta confirmed: %f", sqrt( (mu_1-mu_2)' * beta ))
norm((mu_1-mu_2))
tic




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
    clustering_acc_mat(j) = iterative_kmeans_spectral_init_covar_ver_02_06_24(x_noisy, Sigma, K, 10, cluster_true, 'spec', false, 'basic');
    acc_so_far =  clustering_acc_mat(1:j);
    fprintf( "mean acc so far: %f\n",  mean( acc_so_far ) );

    toc
        % iterate        
end

csvwrite(path_result, clustering_acc_mat)