pc = parallel.cluster.Local;
job_folder = fullfile('/scratch1/',getenv('USER'),getenv('SLURM_JOB_ID'));
mkdir(job_folder);
set(pc,'JobStorageLocation',job_folder);
ncores = str2num(getenv('SLURM_CPUS_PER_TASK')) - 1;
pool = parpool(pc,ncores)

rho = 20;
Delta = 5
p = 400

cluster_home = '/home1/jongminm'
project_name = 'sparse_kmeans'
path_project = strcat(cluster_home, '/',project_name);
addpath(genpath( path_project ))
meeting_date = '17_03_08_2024'
experiment_name = 'ISEE_noisy'
path_result = strcat(path_project, '/', meeting_date, '/', experiment_name, 'result')
try_name = strcat('rho', string(rho), '_Delta', string(Delta), '_p', string(p))
path_file_noext = strcat(path_result, '/', try_name)

path_result = strcat(path_file_noext, '.csv')
path_normfromat= strcat( path_file_noext, '_normfromat.csv')
path_suppdiff= strcat( path_file_noext, '_suppdiff.csv')
path_falsediscov= strcat( path_file_noext, '_falsediscov.csv')
path_truediscov= strcat( path_file_noext, '_truediscov.csv')
path_falsediscovtop5= strcat( path_file_noext, '_falsediscovtop5.csv')
path_omegaesttime= strcat( path_file_noext, '_omegaesttime.csv')
path_xtildeesttime= strcat( path_file_noext, '_xtildeesttime.csv')
path_sdpsolvetime= strcat( path_file_noext, '_sdpsolvetime.csv')


rho = rho /100





s = 10;
n_rep = 2;


n=500;
K=2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
n_iter = 6; 










Omega = eye(p) + diag(rho*ones(p-1,1), 1) + diag(rho*ones(p-1,1), -1);




Sigma = inv(Omega);
M = Delta/2/ sqrt( sum( Sigma(1:s,1:s),"all") )
sparse_mean = [repelem(1,s), repelem(0,p-s)]'; %column vector
mu_0_tilde =  M * sparse_mean;
mu_0 = Sigma*mu_0_tilde;
mu_1 = -mu_0;
mu_2 = mu_0;

beta = Omega * (mu_1-mu_2);
fprintf( "delta confirmed: %f", sqrt( (mu_1-mu_2)' * beta ))
norm((mu_1-mu_2))





mu_1_mat = repmat(mu_1,  1, n/2); %each column is one observation
mu_2_mat = repmat(mu_2, 1, n/2);%each column is one observation
x_noiseless = [ mu_1_mat  mu_2_mat ];%each column is one observation 
x_noisy = x_noiseless +  mvnrnd(zeros(p,1), Sigma, n)';


clustering_acc_mat = zeros(n_rep);
diff_x_tilde = zeros(n_rep, n_iter);
diff_omega_diag = zeros(n_rep, n_iter);
false_discov = zeros(n_rep, n_iter);
true_discov = zeros(n_rep, n_iter);
false_discov_top5 = repmat("0", [n_rep, n_iter]);
omega_est_time = zeros(n_rep, n_iter);
sdp_solve_time = zeros(n_rep, n_iter);

for j = 1:n_rep
    fprintf("iteration: (%i)th \n\n", j)
    rng(j);

    %data generation
    x_noisy = x_noiseless +  mvnrnd(zeros(p,1), Sigma, n)';%each column is one observation
    

    [clustering_acc_mat(j), diff_x_tilde(j,:), diff_omega_diag(j,:), false_discov(j,:), true_discov(j,:), false_discov_top5(j,:), omega_est_time(j,:), sdp_solve_time(j,:)]= iterative_kmeans_spectral_init_ISEE(x_noisy, K,n_iter, Omega, s, cluster_true, 'hc', true, 'basic');
    acc_so_far =  clustering_acc_mat(1:j);
    fprintf( "mean acc so far: %f\n",  mean( acc_so_far ) );


        % iterate        
end

csvwrite(path_result, clustering_acc_mat)
csvwrite(path_normfromat, norm_fro_mat)
csvwrite(path_suppdiff, supp_diff)
csvwrite(path_falsediscov, false_discov)
csvwrite(path_truediscov, true_discov)
csvwrite(path_falsediscovtop5, false_discov_top5)
csvwrite(path_omegaesttime, omega_est_time)
csvwrite(path_xtildeesttime, x_tilde_est_time)
csvwrite(sdp_solve_time, path_sdpsolvetime)

