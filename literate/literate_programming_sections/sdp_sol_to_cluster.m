%% sdp_sol_to_cluster
% @export
function cluster_est = sdp_sol_to_cluster(Z_opt, K)
    [U_sdp, ~, ~] = svd(Z_opt); % extract the left singular vectors
    U_top_K = U_sdp(:, 1:K); % columns are singular vectors. extract to K. thus U_top_K is n x K (n data points, K features)
    cluster_labels = kmeans(U_top_K, K, 'Replicates', 10, 'MaxIter', 500); % Added options for robustness
    % Return cluster assignments as a row vector
    cluster_est = cluster_labels';
end
