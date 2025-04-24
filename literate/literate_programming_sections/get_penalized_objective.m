function obj = get_penalized_objective(X, G)
%% get_penalized_objective
% @export
% 
% *Inputs:* 
%% 
% * X: p x n data matrix (usually a truncated matrix, where p is |S| where S 
% is selected variables) )
% * G: 1 x n vector of cluster labels in {1,...,K}
% Computes the penalized objective combining the profile likelihood 
% and squared L2 distance between cluster means.
%
% Inputs:
%   X : p x n data matrix
%   G : 1 x n vector of cluster labels
%
% Output:
%   obj : scalar value of penalized objective
    [p, n] = size(X);
    n1 = sum(G==1);
    n2 = sum(G==2);
    sd_noise_entry = (n / (n1*n2));
    % Reuse core likelihood component
    lik_obj = get_likelihood_objective(X, G);    
    % Compute cluster means
    cluster_mean_one = mean(X(:, G == 1), 2);  % p x 1
    cluster_mean_two = mean(X(:, G == 2), 2);  % p x 1
    % Compute squared L2 distance between cluster means
    diff = cluster_mean_one - cluster_mean_two;
    penalty = n * sum(diff .^ 2) / sd_noise_entry;
    % Combine likelihood and penalty
    obj = lik_obj + penalty;
end
