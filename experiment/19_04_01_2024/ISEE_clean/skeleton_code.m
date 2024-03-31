
% data paramters

init_method = 'spec'

zero_mean = zeros(p,1);
for jj = 1:4
    rep = (ii-1)*4+jj
    rng(rep)
    x_noisy = x_noiseless +  mvnrnd(zero_mean, Sigma, n)';%data generation. each column is one observation

    fprintf("replication: (%i)th \n\n", rep)
    iterative_kmeans_learner = iterative_kmeans(x_noisy, @data_gaussian_ISEE_clean, number_cluster, omega_sparsity, init_method);
    iterative_kmeans_learner.learn(n_iter_max);
    
    database_subtable = iterative_kmeans_learner.get_database_subtable(rep, Delta, rho, s, cluster_true, Omega)
    conn=sqlite('/home1/jongminm/sparse_kmeans/sparse_kmeans.db');
    pause(2);
    sqlwrite(conn, table_name, database_subtable)
    pause(2);
    close(conn)
end


