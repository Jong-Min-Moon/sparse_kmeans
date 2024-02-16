pkl_path = '/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/rho45_Delta3_p100.pkl'
mat_path = '/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/rho45_Delta3_p100.mat'
ebic_path = '/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/rho45_Delta3_p100.py'
rho = 45;
p = 100
Delta = 3
path_result = '/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/result/rho45_Delta3_p100.csv'
path_normfromat= '/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/result/rho45_Delta3_p100_normfromat.csv'
path_suppdiff= '/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/result/rho45_Delta3_p100_suppdiff.csv'
path_falsediscov= '/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/result/rho45_Delta3_p100_falsediscov.csv'
path_truediscov= '/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/result/rho45_Delta3_p100_truediscov.csv'
path_falsediscovtop5= '/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/result/rho45_Delta3_p100_falsediscovtop5.csv'
path_omegaesttime= '/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/result/rho45_Delta3_p100_omegaesttime.csv'
path_xtildeesttime= '/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/result/rho45_Delta3_p100_xtildeesttime.csv'
path_sdpsolvetime= '/mnt/nas/users/user213/sparse_kmeans/experiment/14_02_16_2024/glasso/result/rho45_Delta3_p100_sdpsolvetime.csv'

%p=
%Delta=
%rho = 
rho = rho /100
addpath(genpath('/mnt/nas/users/user213/sparse_kmeans'))
feature("numcores")
maxNumCompThreads(2);




s = 10;
n_rep = 100;


n=500;
K=2;
rounding = 1e-4;
cluster_true = [repelem(1,n/2), repelem(-1,n/2)];
n_iter = 6; 










Omega = zeros([p,p]);

for j=1:p
    for l = 1:p
        if j==l
            Omega(j,l) = 1;
        elseif abs(j-l) ==1
            Omega(j,l) = rho;
        end
    end
end



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
norm_fro_mat = zeros(n_rep, n_iter);
supp_diff = zeros(n_rep, n_iter);
false_discov = zeros(n_rep, n_iter);
true_discov = zeros(n_rep, n_iter);
false_discov_top5 = repmat("0", [n_rep, n_iter]);
omega_est_time = zeros(n_rep, n_iter);
x_tilde_est_time = zeros(n_rep, n_iter);
sdp_solve_time = zeros(n_rep, n_iter);

for j = 1:n_rep
    fprintf("iteration: (%i)th \n\n", j)
    rng(j);

    %data generation
    x_noisy = x_noiseless +  mvnrnd(zeros(p,1), Sigma, n)';%each column is one observation
    [clustering_acc_mat(j), norm_fro_mat(j,:), supp_diff(j,:), false_discov(j,:), true_discov(j,:), false_discov_top5(j,:), omega_est_time(j,:), x_tilde_est_time(j,:), sdp_solve_time(j,:)]= iterative_kmeans_spectral_init_glasso(x_noisy, K,n_iter, Omega, s, cluster_true, 'hc', true, 'basic', "/opt/miniconda/bin", "/mnt/nas/users/user213/.conda/envs/kmeans", pkl_path, mat_path, ebic_path);
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

