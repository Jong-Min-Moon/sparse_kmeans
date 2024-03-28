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


rho = 45;
n_iter = 30;
p = 50
Delta = 6
s = 10
n = 500
ii = 2
table_name = 'sparse_kmeans_isee_denoise'

conn=sqlite('/home1/jongminm/sparse_kmeans/sparse_kmeans.db')
rho = rho /100


% data paramters
K=2;
Omega_sparsity = 2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
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

table_cell = cell(1,4);
zero_mean = zeros(p,1);
for jj = 1:4
    rep = (ii-1)*4+jj
    rng(rep)
    x_noisy = x_noiseless +  mvnrnd(zero_mean, Sigma, n)';%data generation. each column is one observation

    fprintf("replication: (%i)th \n\n", rep)

    [cluster_est_mat, diff_x_tilde, diff_omega_diag, entries_survived, omega_est_time, sdp_solve_time, obj_prim, obj_dual]= iterative_kmeans_ISEE_denoise(x_noisy, K, n_iter, Omega, Omega_sparsity, 'spec');

    acc_vec = get_acc(cluster_est_mat, cluster_true)
    fprintf( strcat( "acc =", join(repelem("%f ", length(acc_vec))), "\n"),  acc_vec );
    [discov_true_vec, discov_false_vec] = get_discovery(entries_survived, s);
    discov_true_vec
    discov_false_vec

    import java.util.TimeZone 
    nn = now;
    ds = datestr(nn);
    dt = datetime(ds,'TimeZone',char(TimeZone.getDefault().getID()));
    data = table(...
        repelem(rep, n_iter+1)',...
        (0:n_iter)',...
        repelem(Delta, n_iter+1)',...
        repelem(p, n_iter+1)',...
        repelem(rho, n_iter+1)',...
        repelem(s, n_iter+1)',...
        acc_vec,...
        [0;, obj_prim],...
        [0;, obj_dual],...
        [0; discov_true_vec],...
        [0; discov_false_vec],...
        [0; diff_x_tilde],...
        [0; diff_omega_diag],...
        [0; omega_est_time],...
        [0; sdp_solve_time], repelem(dt, n_iter+1)','VariableNames', ...
        ["rep", "iter", "sep", "dim", "rho", "sparsity", "acc", "obj_prim", "obj_dual", "discov_true", "discov_false", "diff_x_tilde", "diff_omega_diag",  "time_isee", "time_SDP", "jobdate"])
        sqlwrite(conn, table_name, data)
end

close(conn)
