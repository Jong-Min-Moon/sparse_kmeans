%% sdp_kmeans
% @export
% Solver: SDPNAL+
function cluster_est = sdp_kmeans(X, K)
% Input validation
if nargin < 2
    error('sdp_kmeans:NotEnoughInputs', 'Two input arguments required: data matrix X and number of clusters K.');
end
if ~ismatrix(X) || ~isnumeric(X)
    error('sdp_kmeans:InvalidX', 'Input X must be a numeric matrix.');
end
if ~isscalar(K) || K <= 1 || K ~= floor(K)
    error('GET_CLUSTER_BY_BY_SDP:InvalidK', 'Number of clusters K must be an integer greater than 1.');
end
[d, n] = size(X); % d is dimension, n is number of data points
if K > n
    error('sdp_kmeans:KExceedsN', 'Number of clusters K cannot exceed the number of data points (%d).', n);
end
 X_scaled = normalize(X');
D =  X_scaled * X_scaled';
Z_opt = kmeans_sdp_pengwei(D, K);
% Check if Z_opt is valid
if isempty(Z_opt) || ~ismatrix(Z_opt) || ~isnumeric(Z_opt)
    error('sdp_kmeans:InvalidZOpt', 'The SDP solver ''kmeans_sdp_pengwei'' returned an invalid or empty solution.');
end
% Perform eigendecomposition on the SDP solution
% The SDP solution Z_opt is often a matrix (e.g., n x n) from which
% eigenvectors are extracted for spectral clustering.
% Assuming Z_opt is an n x n matrix where n is the number of data points.
if size(Z_opt, 1) ~= n || size(Z_opt, 2) ~= n
    warning('sdp_kmeans:ZOptDimensionMismatch', ...
        'Expected Z_opt to be an %d x %d matrix, but got %d x %d. Proceeding with SVD, but results might be unexpected.', ...
        n, n, size(Z_opt, 1), size(Z_opt, 2));
end
cluster_est = sdp_sol_to_cluster(Z_opt, K);
end
 
%% 
%%  
