

% Determine number of cores for parallel pool
ncores = str2num(getenv('SLURM_CPUS_PER_TASK')) - 1;
pool = parpool(pc, ncores);

 
db_dir = '/home1/jongminm/sparse_kmeans/sparse_kmeans.db';

% Add sparse_kmeans path
addpath(genpath('/home1/jongminm/sparse_kmeans'));

% Define the file paths
file_x = "leuk_x.txt";
x = readmatrix(file_x);
disp(['Size of x: ' num2str(size(x))])
x = 2.^x;

file_y = "leuk_y.txt";
y = readmatrix(file_y);
y= y' +1
disp(['Size of colon_y: ' num2str(size(y))]);
 
 
   

 %0.87 vs 0.91        
 
    
n_rep=30;
n_subsample = 50
acc = 0;
acc_ours = 0;
gen = data_generator_subsample(x, y);
for i = 1:n_rep
    [x_new, y_new] = gen.get_data(n_subsample, i);
    ISEE_kmeans_clean_simul(x_new, 2, 200, true, 10, 5, 0.01, db_dir, 'real', 0, 'leuk', 0, y_new);
 
end
