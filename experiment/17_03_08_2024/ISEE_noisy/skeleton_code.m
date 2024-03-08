
rho = rho /100

% data paramters
K=2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
n_iter = 10; 
Omega = eye(p) + diag(rho*ones(p-1,1), 1) + diag(rho*ones(p-1,1), -1);
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

mu_1_mat = repmat(mu_1,  1, n/2); %each column is one observation
mu_2_mat = repmat(mu_2, 1, n/2);%each column is one observation
x_noiseless = [ mu_1_mat  mu_2_mat ];%each column is one observation 


rng(rep)

fprintf("replication: (%i)th \n\n", rep)

%data generation
x_noisy = x_noiseless +  mvnrnd(zeros(p,1), Sigma, n)';%each column is one observation
    

[clustering_acc, diff_x_tilde, diff_omega_diag, false_discov, true_discov, false_discov_top5, omega_est_time, sdp_solve_time]= iterative_kmeans_spectral_init_ISEE_hpc(x_noisy, K,n_iter, Omega, s, cluster_true, 'hc', rep, dcon, true);
    acc_so_far =  clustering_acc_mat(1:j);
    fprintf( "mean acc so far: %f\n",  mean( acc_so_far ) );


        % iterate        
end

csvwrite(path_result, clustering_acc_mat)
csvwrite(path_normfromat, norm_fro_mat)
csvwrite(path_suppdiff, supp_diff)
csvwrite(path_falsediscov, false_discov)
csvwrite(path_truediscov, true_discov)
csvwrite(path_falsediscovtop5, false_discov_top5)
csvwrite(path_omegaesttime, omega_est_time)
csvwrite(path_xtildeesttime, x_tilde_est_time)
csvwrite(sdp_solve_time, path_sdpsolvetime)

