
% data paramters
table_name = sparse_kmeans_isee_clean;
db_dir = "ssh://usc_proxy/home1/jongminm/sparse_kmeans/sparse_kmeans.db"
ii=20;
separation=4;
dimension=100;
sample_size=500;
for jj = 1:4
    experimenter = block_iteration_for_server(table_name, db_dir, 1:10, separation, dimension, 0.45, sample_size)
    database_subtable = bifs.run_one_iteration(ii, jj)
    experimenter.save_into_database(database_subtable)



    
    rep = (ii-1)*4+jj
    rng(rep)
    x_noisy = x_noiseless +  mvnrnd(zero_mean, Sigma, n)';%data generation. each column is one observation

    fprintf("replication: (%i)th \n\n", rep)
    iterative_kmeans_learner = iterative_kmeans(x_noisy, @data_gaussian_ISEE_clean, number_cluster, omega_sparsity, init_method);
    iterative_kmeans_learner.learn(n_iter_max);
    
    database_subtable = iterative_kmeans_learner.get_database_subtable(rep, Delta, rho, s, cluster_true, Omega);
    database_subtable(:,1:12)
    random_seconds = randi([4 32],1);
    pause(random_seconds)
    conn=sqlite('/home1/jongminm/sparse_kmeans/sparse_kmeans.db');
    pause(2);
    sqlwrite(conn, table_name, database_subtable)
    pause(2);
    close(conn)
end


