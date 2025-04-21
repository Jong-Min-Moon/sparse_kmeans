function obj = get_penalized_objective(X, G)
%% get_penalized_objective
% @export
% 
% *Inputs:* 
%% 
% * X: p x n data matrix (usually a truncated matrix, where p is |S| where S 
% is selected variables) )
% * G: 1 x n vector of cluster labels in {1,...,K}
% Computes the full profile likelihood objective
% X: p x n data matrix
% G: 1 x n vector of cluster labels
[p,n] = size(X);
lik_obj = get_likelihood_objective(X, G);      % reuse core SDP component
obj = lik_obj + n*p;
end
