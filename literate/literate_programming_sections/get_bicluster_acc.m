function acc = get_bicluster_acc(cluster_est, cluster_true)
%% get_bicluster_acc
% @export
% 
% computes the clustering accuracy between two label vectors, accounting for 
% label permutations
% 
% Inputs:
%% 
% * cluster_est: estimated labels (vector of 1s and 2s)
% * cluster_true: ground truth labels (vector of 1s and 2s)
%% 
% Output:
%% 
% * acc: clustering accuracy (between 0 and 1)
    % Ensure column vectors
    cluster_est = cluster_est(:);
    cluster_true = cluster_true(:);
    if length(cluster_est) ~= length(cluster_true)
        error('Input vectors must be the same length.');
    end
    % Case 1: no permutation
    acc1 = sum(cluster_est == cluster_true);
    % Case 2: flip labels in estimated
    cluster_est_flipped = 3 - cluster_est; % flip 1<->2
    acc2 = sum(cluster_est_flipped == cluster_true);
    % Choose maximum accuracy
    acc = max(acc1, acc2) / length(cluster_true);
end
