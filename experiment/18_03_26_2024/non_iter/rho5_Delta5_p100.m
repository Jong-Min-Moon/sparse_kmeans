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


rho = 5;
n_iter = 30;
p = 100
Delta = 5
s = 10
n = 500
table_name = 'sparse_kmeans_non_iter'
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




    


for rep = 1:100
    rng(rep)
    x_noisy = x_noiseless +  mvnrnd(zero_mean, Sigma, n)';%data generation. each column is one observation

    fprintf("replication: (%i)th \n\n", rep)
    [cluster_est, diff_x_tilde, diff_omega_diag, omega_est_time, sdp_solve_time, obj_val_prim, obj_val_dual] = non_iter_kmeans(x_noisy, K,  Omega,  'spec');
    acc_vec = get_acc(cluster_est, cluster_true);
    fprintf( strcat( "acc =", join(repelem("%f ", length(acc_vec))), "\n"),  acc_vec );



    import java.util.TimeZone 
    nn = now;
    ds = datestr(nn);
    dt = datetime(ds,'TimeZone',char(TimeZone.getDefault().getID()));
    data = table(...
        rep,...
        Delta,...
        p,...
        rho,...
        s,...
        acc_vec,...
        obj_val_prim,...
        obj_val_dual,...
        diff_x_tilde,...
        diff_omega_diag,...
        omega_est_time,...
        sdp_solve_time,...
        dt,'VariableNames', ...
        ["rep", "sep", "dim", "rho", "sparsity", "acc", "obj_prim", "obj_dual", "diff_x_tilde", "diff_omega_diag",  "time_isee", "time_SDP", "jobdate"])
        waittime = randi([5,20]);
        pause(waittime);
        conn=sqlite('/home1/jongminm/sparse_kmeans/sparse_kmeans.db')
        pause(2);
        sqlwrite(conn, table_name, data)
        pause(2);
        close(conn)
end


