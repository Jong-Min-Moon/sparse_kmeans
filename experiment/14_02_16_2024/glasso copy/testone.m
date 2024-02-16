rho = 2;
p = 100
Delta = 4
%path_result = '/mnt/nas/users/user213/sparse_kmeans/experiment/13_02_09_2024/AR1/result/rho2_Delta4_p800.csv'

%p=
%Delta=
%rho = 
rho = rho /10
%addpath(genpath('/mnt/nas/users/user213/sparse_kmeans'))
feature("numcores")




s = 10;
n_rep = 1;


n=500;
K=2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
n_iter = 5; 



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


Sigma = inv(Omega);
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
    clustering_acc_mat(j) = iterative_kmeans_spectral_init_glasso(x_noisy, K,n_iter,s, cluster_true, 'hc', true, 'basic');
    acc_so_far =  clustering_acc_mat(1:j);
    fprintf( "mean acc so far: %f\n",  mean( acc_so_far ) );

    toc
        % iterate        
end

%csvwrite(path_result, clustering_acc_mat)
