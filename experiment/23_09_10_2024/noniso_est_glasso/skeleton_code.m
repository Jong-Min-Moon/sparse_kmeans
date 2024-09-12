rho=rho/100
data_obj = @data_gaussian_glasso
matrix_sparsity =2 ;
init_method = "spec";
full_run = false
flip=false
for jj = 1:4
    experimenter = block_replication_for_server(table_name, db_dir, 1:10, separation, dimension, rho, sample_size, n_iter_max, full_run, init_method, matrix_sparsity, data_obj, flip);
    database_subtable = experimenter.run_one_replication(ii, jj)
    experimenter.save_into_database(database_subtable)
end










