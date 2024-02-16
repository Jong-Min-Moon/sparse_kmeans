<<<<<<<< HEAD:experiment/14_02_16_2024/glasso copy/testone.m
rho = 2;
p = 100
Delta = 4
%path_result = '/mnt/nas/users/user213/sparse_kmeans/experiment/13_02_09_2024/AR1/result/rho2_Delta4_p800.csv'
========
rho = 5;
p = 1500
Delta = 4
path_result = '/mnt/nas/users/user213/sparse_kmeans/experiment/13_02_09_2024/AR1/result/rho5_Delta4_p1500.csv'
>>>>>>>> ae554d7645a21218e49e9cddc69cca5c92bed035:experiment/13_02_09_2024/AR1/rho5_Delta4_p1500.m

%p=
%Delta=
%rho = 
rho = rho /10
<<<<<<<< HEAD:experiment/14_02_16_2024/glasso copy/testone.m
%addpath(genpath('/mnt/nas/users/user213/sparse_kmeans'))
feature("numcores")
========
addpath(genpath('/mnt/nas/users/user213/sparse_kmeans'))
feature("numcores")
maxNumCompThreads(1);
>>>>>>>> ae554d7645a21218e49e9cddc69cca5c92bed035:experiment/13_02_09_2024/AR1/rho5_Delta4_p1500.m




s = 10;
<<<<<<<< HEAD:experiment/14_02_16_2024/glasso copy/testone.m
n_rep = 1;
========
n_rep = 100;
>>>>>>>> ae554d7645a21218e49e9cddc69cca5c92bed035:experiment/13_02_09_2024/AR1/rho5_Delta4_p1500.m


n=200;
K=2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
<<<<<<<< HEAD:experiment/14_02_16_2024/glasso copy/testone.m
n_iter = 5; 
========
n_iter = 10; 
>>>>>>>> ae554d7645a21218e49e9cddc69cca5c92bed035:experiment/13_02_09_2024/AR1/rho5_Delta4_p1500.m



clustering_acc_mat = repelem(0, n_rep );






<<<<<<<< HEAD:experiment/14_02_16_2024/glasso copy/testone.m
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
========
Sigma = zeros([p,p]);

for j=1:p
    for l = 1:p
        Sigma(j,l) = rho^(abs(j-l));
    end
end

>>>>>>>> ae554d7645a21218e49e9cddc69cca5c92bed035:experiment/13_02_09_2024/AR1/rho5_Delta4_p1500.m


M = Delta/2/ sqrt( sum( Sigma(1:s,1:s),"all") )
sparse_mean = [repelem(1,s), repelem(0,p-s)]'; %column vector
mu_0_tilde =  M * sparse_mean;
mu_0 = Sigma*mu_0_tilde;
mu_1 = -mu_0;
mu_2 = mu_0;

beta = linsolve(Sigma, (mu_1-mu_2));
fprintf( "delta confirmed: %f", sqrt( (mu_1-mu_2)' * beta ))
norm((mu_1-mu_2))
tic

% norm(mu_1 - mu_2)


% parallel for loop

mu_1_mat = repmat(mu_1,  1, n/2); %each column is one observation
mu_2_mat = repmat(mu_2, 1, n/2);%each column is one observation
x_noiseless = [ mu_1_mat  mu_2_mat ];%each column is one observation 

for j = 1:n_rep
    fprintf("iteration: (%i)th \n\n", j)
    rng(j+100)
    tic
    %data generation
    
    x_noisy = x_noiseless +  mvnrnd(zeros(p,1), Sigma, n)';%each column is one observation
<<<<<<<< HEAD:experiment/14_02_16_2024/glasso copy/testone.m
    clustering_acc_mat(j) = iterative_kmeans_spectral_init_glasso(x_noisy, K,n_iter,s, cluster_true, 'hc', true, 'basic');
========
    clustering_acc_mat(j) = iterative_kmeans_spectral_init_covar_ver_02_06_24(x_noisy, Sigma, K, 10, cluster_true, 'spec', false, 'basic');
>>>>>>>> ae554d7645a21218e49e9cddc69cca5c92bed035:experiment/13_02_09_2024/AR1/rho5_Delta4_p1500.m
    acc_so_far =  clustering_acc_mat(1:j);
    fprintf( "mean acc so far: %f\n",  mean( acc_so_far ) );

    toc
        % iterate        
end

<<<<<<<< HEAD:experiment/14_02_16_2024/glasso copy/testone.m
%csvwrite(path_result, clustering_acc_mat)
========
csvwrite(path_result, clustering_acc_mat)
>>>>>>>> ae554d7645a21218e49e9cddc69cca5c92bed035:experiment/13_02_09_2024/AR1/rho5_Delta4_p1500.m
