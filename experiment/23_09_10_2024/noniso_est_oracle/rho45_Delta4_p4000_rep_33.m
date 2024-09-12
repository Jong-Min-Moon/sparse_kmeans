addpath(genpath('/home1/jongminm/sparse_kmeans'));
pc = parallel.cluster.Local;
job_folder = fullfile('/scratch1/',getenv('USER'),getenv('SLURM_JOB_ID'));
mkdir(job_folder);
set(pc,'JobStorageLocation',job_folder);
ncores = str2num(getenv('SLURM_CPUS_PER_TASK')) - 1;
pool = parpool(pc,ncores)
%
%
%


n_iter_max = 100;
ii = 33;
rho = 45;
dimension = 4000
separation = 4
sample_size = 
table_name = 'noniso_est_oracle'
db_dir = '/home1/jongminm/sparse_kmeans/sparse_kmeans.db'
rho=rho/100
data_obj = "oracle";
init_method = "spec";
matrix_sparsity =2 ;
full_run = false
flip=false
for jj = 1:4
    experimenter = block_replication_for_server(table_name, db_dir, 1:10, separation, dimension, rho, sample_size, n_iter_max, full_run, init_method, matrix_sparsity, data_obj, flip);
    database_subtable = experimenter.run_one_replication(ii, jj)
    experimenter.save_into_database(database_subtable)
end


