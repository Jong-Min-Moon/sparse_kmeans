pc = parallel.cluster.Local
job_folder = fullfile('/home1/jongminm/.matlab/local_cluster_jobs/R2022a',getenv('SLURM_JOB_ID'))
if ~exist(job_folder, 'dir')
    mkdir(job_folder);
end
set(pc,'JobStorageLocation',job_folder);
ncores = str2num(getenv('SLURM_CPUS_PER_TASK')) - 1
pool = parpool(pc,ncores)

p = 400;
rep = 10;
    sep=4;
    n = 500;
    addpath(genpath('/home1/jongminm/sparse_kmeans'))
  
    % Set database and table
    table_name = 'isee_new';
    db_dir = '/home1/jongminm/sparse_kmeans/sparse_kmeans.db';
    
    % Model setup
    model = 'chain45';
    cluster_1_ratio = 0.5;
    
    % Generate data
    [data, label_true, mu1, mu2, sep, ~, beta_star]  = generate_gaussian_data(n, p, 4, model, rep, cluster_1_ratio);
    data = data';
    label_true = label_true';
    %                                 T   parallel  loop_detect_start  window_size    0.min_delta     
    ISEE_kmeans_clean_simul(data, 2, 100, true,       10,              5,               0.01, db_dir, table_name, rep, model, sep, label_true)    
    % Run our method    
    % Evaluate clustering accuracy
    delete(pool)
