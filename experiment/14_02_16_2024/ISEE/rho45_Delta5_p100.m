rho = 45;
Delta = 5
p = 100


%p=
%Delta=
%rho = 
rho = rho /100





s = 10;
n_rep = 100;


n=500;
K=2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
n_iter = 6; 










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
x_noisy = x_noiseless +  mvnrnd(zeros(p,1), Sigma, n)';


clustering_acc_mat = zeros(n_rep);
diff_x_tilde = zeros(n_rep, n_iter);
diff_omega_diag = zeros(n_rep, n_iter);
false_discov = zeros(n_rep, n_iter);
true_discov = zeros(n_rep, n_iter);
false_discov_top5 = repmat("0", [n_rep, n_iter]);
omega_est_time = zeros(n_rep, n_iter);
sdp_solve_time = zeros(n_rep, n_iter);

for j = 1:n_rep
    fprintf("iteration: (%i)th \n\n", j)
    rng(j);

    %data generation
    x_noisy = x_noiseless +  mvnrnd(zeros(p,1), Sigma, n)';%each column is one observation
    

    [clustering_acc_mat(j), diff_x_tilde(j,:), diff_omega_diag(j,:), false_discov(j,:), true_discov(j,:), false_discov_top5(j,:), omega_est_time(j,:), sdp_solve_time(j,:)]= iterative_kmeans_spectral_init_ISEE(x_noisy, K,n_iter, Omega, s, cluster_true, 'hc', true, 'basic');
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

