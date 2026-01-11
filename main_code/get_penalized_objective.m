%% get_penalized_objective
% @export
% 
% *Inputs:* 
%% 
% * X: p x n data matrix (usually a truncated matrix, where p is |S| where S 
% is selected variables) )
% * G: 1 x n vector of cluster labels in {1,...,K}
function obj = get_penalized_objective(X, G)
% Computes the penalized objective combining the profile likelihood 
% and squared Frobenius norm of the row-centered data matrix.
%
% Inputs:
%   X : p x n data matrix
%   G : 1 x n vector of cluster labels
%
% Output:
%   obj : scalar value of penalized objective
    [p, n] = size(X);
    
    % Core likelihood component
    lik_obj = get_likelihood_objective(X, G);    
    % Compute penalty using variance
    penalty = 2 * (n - 1) * sum(var(X, 0, 2)); 
    % Combine likelihood and penalty
    obj = lik_obj + penalty;
end
