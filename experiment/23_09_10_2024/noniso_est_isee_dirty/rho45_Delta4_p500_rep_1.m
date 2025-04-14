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
ii = 1;
rho = 45;
dimension = 400
separation = 4
sample_size = 500
table_name = 'noniso_est_isee_dirty'
db_dir = '/home1/jongminm/sparse_kmeans/sparse_kmeans.db'
rho=rho/100
data_obj = @data_gaussian_ISEE_dirty
matrix_sparsity =2 ;
init_method = "spec";
full_run = false
flip=false

data_generator = sparse_symmetric_data_generator(1:10, separation, dimension, 2, rho, flip)
cluster_true = [repelem(1,sample_size/2), repelem(2,sample_size/2)];    
zero_mean = zeros(dimension,1);
x_noiseless = data_generator.get_noiseless_data(sample_size);
rng(2)
x_noisy = x_noiseless +  mvnrnd(zero_mean, data_generator.covariance_matrix, sample_size)';%data generation. each column is one observation
cluster_estimte = ISEE_kmeans_noisy(x_noisy, 2, 100, true)
acc = get_bicluster_acc(cluster_estimte, cluster_true)


