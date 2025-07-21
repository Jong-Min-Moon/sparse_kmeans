function acc = get_bicluster_accuracy(cluster_est, cluster_true)
%% get_bicluster_accuracy
% @export
%% 
% * Computes accuracy for a two-cluster assignment.
% * acc = GET_CLUSTER_ACCURACY(cluster_est, cluster_true) returns the proportion 
% of correctly assigned labels, accounting for label switching.
%% 
% *Inputs*
%% 
% * cluster_est: n array of 1 and 2
% * cluster_true: n array of 1 and 2
%% 
% Outputs
%% 
% * acc: ratio of correctly clustered observations
%% 
% 
    % Ensure both inputs are vectors
    if ~isvector(cluster_est) || ~isvector(cluster_true)
        error('Both inputs must be vectors.');
    end
    % If one is row and one is column, transpose the row vector
    if size(cluster_est, 1) == 1 && size(cluster_true, 1) > 1
        cluster_est = cluster_est';
    elseif size(cluster_true, 1) == 1 && size(cluster_est, 1) > 1
        cluster_true = cluster_true';
    end
    if length(cluster_est) ~= length(cluster_true)
        error('Input vectors must be the same length.');
    end
    % Compute accuracy under both labelings
    match1 = sum(cluster_est == cluster_true);
    match2 = sum(cluster_est == (3 - cluster_true));  % flips 1<->2
    acc = max(match1, match2) / length(cluster_true);
end
%% 
% 
