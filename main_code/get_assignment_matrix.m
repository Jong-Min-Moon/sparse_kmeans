%% get_assignment_matrix
% @export
function A_double = get_assignment_matrix(n, K, cluster_labels)
% GET_ASSIGNMENT_MATRIX Creates a binary assignment matrix from cluster labels.
%
%   A_double = GET_ASSIGNMENT_MATRIX(n, K, cluster_labels) creates a binary
%   assignment matrix where A(j, k) = 1 if data point j belongs to cluster k,
%   and 0 otherwise. This function assumes cluster labels are integers from 1 to K.
%
%   Inputs:
%     n              - (numeric) Total number of data points.
%     K              - (numeric) Total number of clusters.
%     cluster_labels - A vector of cluster assignments for each data point.
%                      Expected to contain integer values from 1 to K.
%
%   Outputs:
%     A_double       - A (n x K) double matrix representing the
%                      binary assignment of data points to clusters.
%
%   Example:
%     n_points = 100;
%     n_clusters = 3;
%     labels = randi(n_clusters, n_points, 1); % Example labels
%     assignment_mat = get_assignment_matrix(n_points, n_clusters, labels);
%     disp(size(assignment_mat)); % Should be [100 3]
% Input validation
if ~isscalar(n) || ~isnumeric(n) || n < 1 || n ~= floor(n)
    error('get_assignment_matrix:InvalidN', 'Input ''n'' (number of data points) must be a positive integer scalar.');
end
if ~isscalar(K) || ~isnumeric(K) || K < 1 || K ~= floor(K)
    error('get_assignment_matrix:InvalidK', 'Input ''K'' (number of clusters) must be a positive integer scalar.');
end
if ~isnumeric(cluster_labels) || ~isvector(cluster_labels) || any(cluster_labels < 1) || any(cluster_labels > K) || any(cluster_labels ~= floor(cluster_labels))
    error('get_assignment_matrix:InvalidClusterLabels', 'cluster_labels must be a numeric vector of integers from 1 to K.');
end
if length(cluster_labels) ~= n
    error('get_assignment_matrix:DimensionMismatch', 'Length of cluster_labels (%d) must match n (%d).', length(cluster_labels), n);
end
% Initialize A_double as a logical matrix for efficiency
A_double = false(n, K); % A_double will be (num_data_points x K)
% Vectorized way to populate A_double based on cluster_labels
% This works if cluster_labels contains integer IDs from 1 to K
% Ensure cluster_labels is a column vector for sub2ind
linear_indices = sub2ind(size(A_double), (1:n)', cluster_labels(:));
A_double(linear_indices) = true;
% Convert to double for matrix multiplication (as required by your reward function)
A_double = double(A_double);
end
