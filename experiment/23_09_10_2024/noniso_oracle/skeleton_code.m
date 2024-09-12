rho=rho/100
data_obj = "oracle";
init_method = "spec";
matrix_sparsity ="nonsparse" ;
full_run = false
flip=true
window_size_half=2;
loop_start=8
for jj = 1:4
    experimenter = block_replication_for_server(table_name, db_dir, 1:10, separation, dimension, rho, sample_size, n_iter_max, full_run, init_method, matrix_sparsity, data_obj, flip,window_size_half,loop_start)
    database_subtable = experimenter.run_one_replication(ii, jj)
    experimenter.save_into_database(database_subtable)
end


