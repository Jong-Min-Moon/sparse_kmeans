rho=rho/100
data_obj = @data_gaussian;
init_method = "sdp";
matrix_sparsity =0 ;
full_run = false
for jj = 1:4
    experimenter = block_replication_for_server_ifpca(table_name, db_dir, 1:10, separation, dimension, rho, sample_size, n_iter_max, full_run, init_method, matrix_sparsity, data_obj, false, 100, 100);
    database_subtable = experimenter.run_one_replication(ii, jj)
    experimenter.save_into_database(database_subtable)
end


