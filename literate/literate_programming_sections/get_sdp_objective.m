function obj = get_sdp_objective(X, G)
%% get_sdp_objective
% @export
% 
% Computes the SDP objective 
% 
% *Inputs:* 
%% 
% * X: p x n data matrix (usually a truncated matrix, where p is |S| where S 
% is selected variables) )
% * G: 1 x n vector of cluster labels in {1,...,K}
%% 
% 
% 
% 
A = X' * X;        % n x n Gram matrix
K = max(G);        % number of clusters
obj_sum = 0;
for k = 1:K
    idx = find(G == k);
    A_sub = A(idx, idx);
    obj_sum = obj_sum + sum(A_sub, 'all') / numel(idx);
end
obj = 2 * obj_sum;
end
%% 
% 
