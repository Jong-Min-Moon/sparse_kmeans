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


n_iter_max = 10;
ii = 1;
rho = 45;
dimension = 200
separation = 4
sample_size = 500
table_name = 'noniso_est_isee_conv'
db_dir = '/home1/jongminm/sparse_kmeans/sparse_kmeans.db'
rho=rho/100
matrix_sparsity =2 ;
init_method = "spec";
full_run = false
flip=false
T = readtable('/home1/jongminm/sparse_kmeans/experiment/25_04_03_2025/real/leukemia.x.txt');

% If you want to convert to a matrix (numeric only)
M = table2array(T);
M = M(1:3570,:);
%label_true = [2 2 2 1 2 2 2 2 2 1 1 2 2 2 1 1 1 1 2 2 2 1 2 2 2 1 2 1 1 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 1 2 2 2 2 2 1 2 2 2 1 1 1 2 2 2 2 2 2 2 2 1 2 2 2 1 2 2 1 1 2 2 2 1 2 2]
y = readtable('/home1/jongminm/sparse_kmeans/experiment/25_04_03_2025/real/colon.y.txt')
 y = table2array(y)
 y = y+1
 label_true = y'
data_obj_now = data_gaussian_ISEE_clean(M, rho);
learner = iterative_kmeans(data_obj_now, 2, rho, 'spec');
n_iter = 300
learner.run_iterative_algorithm(n_iter, 1000, 0.01, true, 1000);
acc = learner.evaluate_accuracy(label_true)
disp(acc)
acc(n_iter)
acc(n_iter)
