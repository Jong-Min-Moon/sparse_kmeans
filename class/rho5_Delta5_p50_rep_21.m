
%
%


rho = 5;
n_iter = 5;
p = 50
Delta = 5
s = 10
n = 500
ii = 21

rho = rho /100


% data paramters
number_cluster=2;
omega_sparsity = 2;
init_method = 'spec'

rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(2,n/2)];
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

table_cell = cell(1,4);
zero_mean = zeros(p,1);
for jj = 1:1
    rep = (ii-1)*4+jj
    rng(rep)
    x_noisy = x_noiseless +  mvnrnd(zero_mean, Sigma, n)';%data generation. each column is one observation

    fprintf("replication: (%i)th \n\n", rep)
    iterative_kmeans_learner = iterative_kmeans(x_noisy, @data_gaussian_ISEE_clean, number_cluster, omega_sparsity, init_method);
    iterative_kmeans_learner.learn(n_iter);

    acc_vec = iterative_kmeans_learner.evaluate_accuracy(cluster_true);
    [discov_true_vec, discov_false_vec, survived_indices] = iterative_kmeans_learner.evaluate_discovery(s);
    [diff_x_tilde_fro, diff_x_tilde_op, diff_x_tilde_ellone] = iterative_kmeans_learner.evaluate_innovation_est(Omega);
    fprintf( strcat( "acc =", join(repelem("%f ", length(acc_vec))), "\n"),  acc_vec );
    
    database_subtable = iterative_kmeans_learner.get_database_subtable(rep, Delta, rho, s, cluster_true, Omega)
  
        %sqlwrite(conn, table_name, data)
end

close(conn)
