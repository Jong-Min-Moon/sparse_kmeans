function s_hat = select_variable_ISEE_noisy(mean_now, noise_now, Omega_diag_hat, cluster_est_prev)
%% select_variable_ISEE_noisy
% @export
% 
% Inputs:
%% 
% * mean_now: $p\times n$ matrix of  cluster center part of the innovated data 
% matrix (pre-multiplied by precision matrix), where $p$ is the data dimension 
% and $n$ is the sample size
% * noise_now: $p\times n$ matrix of Gaussian noise part of the data matrix 
% (pre-multiplied by precision matrix), where $p$ is the data dimension and $n$ 
% is the sample size
% * Omega_diag_hat: $p$ vector of diagonal entries of precision matrix
% * cluster_est_prev: $n$ array of positive integers, where n is the sample 
% size. ex. [1 2 1 2 3 4 2 ]
%% 
% Outputs:
%% 
% * s_hat: $p$ boolean vector, where true indicates that variable is selected
    x_tilde_now = mean_now + noise_now;
    p = size(mean_now,1);
    n = size(mean_now,2);
    thres = sqrt(2 * log(p) );
    signal_est_now = mean( x_tilde_now(:, cluster_est_prev==1), 2) - mean( x_tilde_now(:, cluster_est_prev==2), 2);
    n_g1_now = sum(cluster_est_prev == 1);
    n_g2_now = sum(cluster_est_prev == 2);
    abs_diff = abs(signal_est_now)./sqrt(Omega_diag_hat) * sqrt( n_g1_now*n_g2_now/n );
    s_hat = abs_diff > thres;
end
