addpath(genpath('/home1/jongminm/sparse_kmeans'));
n_iter_max = 100;
ii = 5;
rho = 45;
dimension = 250
separation = 9
sample_size = 200
table_name = 'noniso_chime'
db_dir = '/home1/jongminm/sparse_kmeans/sparse_kmeans.db'
rho=rho/100
data_obj = @data_gaussian;
init_method = "sdp";
matrix_sparsity =2 ;
full_run = false
experimenter = block_replication_for_server_chime(table_name, db_dir, 1:10, separation, dimension, rho, sample_size, n_iter_max, full_run, init_method, matrix_sparsity, data_obj, false, 100, 100);
database_subtable = experimenter.run_one_replication(ii, 1)
for jj = 2:200
    experimenter = block_replication_for_server_chime(table_name, db_dir, 1:10, separation, dimension, rho, sample_size, n_iter_max, full_run, init_method, matrix_sparsity, data_obj, false, 100, 100);
    database_subtable = [database_subtable; experimenter.run_one_replication(ii, jj)]
end
experimenter.save_into_database(database_subtable)


