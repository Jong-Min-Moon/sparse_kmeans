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
dimension = 200
separation = 4
sample_size = 500
table_name = 'noniso_est_isee_conv'
db_dir = '/home1/jongminm/sparse_kmeans/sparse_kmeans.db'
rho=rho/100
data_obj = @data_gaussian_ISEE_clean
matrix_sparsity =2 ;
init_method = "spec";
full_run = false
flip=false
T = readtable('/home1/jongminm/sparse_kmeans/experiment/25_04_03_2025/real/my_data.csv');

% If you want to convert to a matrix (numeric only)
M = table2array(T);
label_true = [2 2 2 1 2 2 2 2 2 1 1 2 2 2 1 1 1 1 2 2 2 1 2 2 2 1 2 1 1 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 1 2 2 2 2 2 1 2 2 2 1 1 1 2 2 2 2 2 2 2 2 1 2 2 2 1 2 2 1 1 2 2 2 1 2 2]

data_obj_now = data_gaussian_ISEE_clean(M, rho);
learner = iterative_kmeans(data_obj_now, 2, rho, 'spec');
learner.run_iterative_algorithm(100, 10, 0.01, true, 10);


