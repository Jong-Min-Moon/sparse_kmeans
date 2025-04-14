function Sigma_hat_s_hat_now = get_cov_small(x, cluster_est, s_hat)
%% get_cov_small
% @export
% 
% Inputs:
%% 
% * x: $p\times n$ data matrix, where $p$ is the data dimension and $n$ is the 
% sample size
% * cluster_est_now: $n$ array of positive integers, where n is the sample size. 
% ex. [1 2 1 2 3 4 2 ]
% * s_hat: $p$ boolean vector, where true indicates that variable is selected
%% 
% Outputs:
%% 
% * Sigma_hat_s_hat_now: 
n= size(x,2);
    X_g1_now = x(:, (cluster_est ==  1)); 
    X_g2_now = x(:, (cluster_est ==  2)); 
    X_mean_g1_now = mean(X_g1_now, 2);
    X_mean_g2_now = mean(X_g2_now, 2);
    data_py = [(X_g1_now - X_mean_g1_now) (X_g2_now - X_mean_g2_now)]; % p x n
    data_filtered = data_py(s_hat,:); % p x n
    Sigma_hat_s_hat_now = cov(data_filtered');% input: n x p 
end
