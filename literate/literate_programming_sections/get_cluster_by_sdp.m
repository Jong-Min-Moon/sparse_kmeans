%% get_cluster_by_sdp
% @export
function cluster_est = get_cluster_by_sdp(X, K)
%GET_CLUSTER_BY_SDP Clusters data using SDP relaxation of K-means and spectral clustering.
%
%   cluster_est = GET_CLUSTER_BY_SDP(X, K) performs clustering on the data
%   matrix X using a Semi-Definite Programming (SDP) relaxation of the
%   K-means problem, followed by K-means on the top K eigenvectors of the
%   SDP solution.
%
%   Inputs:
%       X - A numeric matrix where each column is a data point.
%       K - The desired number of clusters (an integer greater than 1).
%
%   Outputs:
%       cluster_est - A row vector containing the cluster assignments for
%                     each data point.
%
%   Example:
%       % Generate some sample data
%       data = [randn(2, 50) + 2, randn(2, 50) - 2];
%       num_clusters = 2;
%       cluster_assignments = get_cluster_by_sdp(data, num_clusters);
%       disp(cluster_assignments);
% Input validation
if nargin < 2
    error('GET_CLUSTER_BY_SDP:NotEnoughInputs', 'Two input arguments required: data matrix X and number of clusters K.');
end
if ~ismatrix(X) || ~isnumeric(X)
    error('GET_CLUSTER_BY_SDP:InvalidX', 'Input X must be a numeric matrix.');
end
if ~isscalar(K) || K <= 1 || K ~= floor(K)
    error('GET_CLUSTER_BY_BY_SDP:InvalidK', 'Number of clusters K must be an integer greater than 1.');
end
[d, n] = size(X); % d is dimension, n is number of data points
if K > n
    error('GET_CLUSTER_BY_SDP:KExceedsN', 'Number of clusters K cannot exceed the number of data points (%d).', n);
end
D = X' * X;
Z_opt = kmeans_sdp_pengwei(D, K);
% Check if Z_opt is valid
if isempty(Z_opt) || ~ismatrix(Z_opt) || ~isnumeric(Z_opt)
    error('GET_CLUSTER_BY_SDP:InvalidZOpt', 'The SDP solver ''kmeans_sdp_pengwei'' returned an invalid or empty solution.');
end
% Perform eigendecomposition on the SDP solution
% The SDP solution Z_opt is often a matrix (e.g., n x n) from which
% eigenvectors are extracted for spectral clustering.
% Assuming Z_opt is an n x n matrix where n is the number of data points.
if size(Z_opt, 1) ~= n || size(Z_opt, 2) ~= n
    warning('GET_CLUSTER_BY_SDP:ZOptDimensionMismatch', ...
        'Expected Z_opt to be an %d x %d matrix, but got %d x %d. Proceeding with SVD, but results might be unexpected.', ...
        n, n, size(Z_opt, 1), size(Z_opt, 2));
end
[U_sdp, S_sdp, V_sdp] = svd(Z_opt); % Using SVD which is robust
% Extract the top K eigenvectors
% These eigenvectors form a new feature space where K-means can be applied.
U_top_K = U_sdp(:, 1:K);
% Apply K-means to the projected data
% The 'kmeans' function expects data points as rows. If U_top_K has
% eigenvectors as columns, we need to transpose it.
% 'kmeans' in MATLAB typically takes data points as rows.
% If U_top_K is n x K (n data points, K eigenvectors), then no transpose
% is needed for kmeans input.
cluster_labels = kmeans(U_top_K, K, 'Replicates', 10, 'MaxIter', 500); % Added options for robustness
% Return cluster assignments as a row vector
cluster_est = cluster_labels';
end
 
