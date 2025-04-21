function obj = get_likelihood_objective(X, G)
%% get_likelihood_objective
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
sdp_obj = get_sdp_objective(X, G);      % reuse core SDP component
frob_norm_sq = norm(X, 'fro')^2;
obj = sdp_obj - 2 * frob_norm_sq;
end
%% 
% 
% 
% 
