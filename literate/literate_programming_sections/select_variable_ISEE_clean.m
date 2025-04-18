function s_hat = select_variable_ISEE_clean(mean_vec, n)
%% select_variable_ISEE_clean
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
    % Validate input dimensions
    [p, col_dim] = size(mean_vec);
    if col_dim ~= 2
        error('mean_vec must be a p-by-2 matrix representing class means.');
    end
    % Estimate sparse support
    mu_diff_hat = mean_vec(:,1) - mean_vec(:,2);
    threshold = 2*sqrt(log(p) * log(n) / n);
    s_hat = abs(mu_diff_hat) > threshold;  % p-dimensional boolean array
    % Print summary
    num_selected = sum(s_hat);
    sum(s_hat(1:10))
    while num_selected == 0
        threshold = threshold /2;
        s_hat = abs(mu_diff_hat) > threshold;
        num_selected = sum(s_hat);
    end
    fprintf('%d out of %d variables selected.\n', num_selected, p);
end
