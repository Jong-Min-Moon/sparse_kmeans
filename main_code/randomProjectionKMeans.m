%% randomProjectionKMeans
% @export
function cluster_est = randomProjectionKMeans(X, k, t)
%randomProjectionKMeans Implements a randomized k-means approximation algorithm.
%   Input:
%     A: n x d matrix (n points, d features)
%     k: Number of clusters
%     epsilon: Error parameter (0 < epsilon < 1/3)
%     gamma_kmeans_algo: A function handle to a gamma-approximation k-means algorithm.
%                        This function should take (data_matrix, num_clusters) as input
%                        and return an indicator matrix (n_rows x k_clusters) where
%                        X(i,j) = 1 if point i belongs to cluster j, and 0 otherwise.
%
%   Output:
%     X_gamma_tilde: Indicator matrix determining a k-partition on the rows of A_tilde.
%                    (n x k matrix)
%
%   Example usage (assuming you have a simple k-means function named 'myKMeans'):
%   % 1. Create dummy data
%   n = 100; % Number of points
%   d = 500; % Number of features
%   A = randn(n, d); % Random data
%   k = 3;   % Number of clusters
%   epsilon = 0.1;
%
%   % 2. Define a simple k-means function (replace with your actual gamma-approx algo)
%   % This example uses MATLAB's built-in kmeans, which returns cluster indices.
%   % We need to convert it to an indicator matrix format.
%   myKMeans = @(data_matrix, num_clusters) convert_labels_to_indicator(...
%                                               kmeans(data_matrix, num_clusters, ...
%                                                      'Replicates', 5, 'MaxIter', 100), ...
%                                               size(data_matrix, 1), num_clusters);
%
%   % Helper function to convert cluster labels to indicator matrix
%   function X_indicator = convert_labels_to_indicator(labels, num_points, num_clusters)
%       X_indicator = zeros(num_points, num_clusters);
%       for i = 1:num_points
%           X_indicator(i, labels(i)) = 1;
%       end
%   end
%
%   % 3. Run the random projection k-means
%   X_gamma_tilde_result = randomProjectionKMeans(A, k, epsilon, myKMeans);
%   disp('Size of the output indicator matrix:');
%   disp(size(X_gamma_tilde_result));
    
    epsilon = 0.1;
    A = X';
    % 1. Set t = Omega(k/epsilon^2)
    % A sufficiently large constant 'c' is needed. Common theoretical bounds
    % might suggest constants around 16 for certain guarantees. Let's use 16 for illustration.
    %c = 0.1;
    %t = ceil(c * k / (epsilon^2)); % Use ceil to ensure t is an integer
    %t=10;
    fprintf('Using t = %d (sketching dimension).\n', t);
    [n, d] = size(A);
    % 2. Compute a random d x t matrix R
    % Rij = +1/sqrt(t) w.p. 1/2, -1/sqrt(t) w.p. 1/2
    R = (rand(d, t) > 0.5) * (1/sqrt(t)) * 2 - (1/sqrt(t));
 
    % 3. Compute the product A_tilde = AR
    A_tilde = A * R;
    X_tilde = A_tilde';
    % 4. Run the gamma-approximation algorithm on A_tilde
    % This step assumes 'gamma_kmeans_algo' is a function handle that
    % can take A_tilde and k as input and return the indicator matrix.
    % You need to provide your specific gamma-approximation k-means algorithm here.
    cluster_est = get_cluster_by_sdp(X_tilde, k);
    fprintf('Random Projection K-Means completed.\n');
end
%% 
% 
