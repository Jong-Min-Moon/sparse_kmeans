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
ii = 44;
rho = 45;
dimension = 400
separation = 4
sample_size = 500
table_name = 'sparse_kmeans_isee_clean'
db_dir = '/home1/jongminm/sparse_kmeans/experiment/22_05_03_2024/db_050324_early_stopping.db'
rho=rho/100
for jj = 1:4
    experimenter = block_replication_for_server(table_name, db_dir, 1:10, separation, dimension, rho, sample_size, n_iter_max, true);
    database_subtable = experimenter.run_one_replication(ii, jj)
    experimenter.save_into_database(database_subtable)
end


