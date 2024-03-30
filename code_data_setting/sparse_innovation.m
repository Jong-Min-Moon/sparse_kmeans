rho = rho /100

% data paramters
K=2;
Omega_sparsity = 2;
number_cluster=2;
omega_sparsity = 2;
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




    


