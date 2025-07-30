%% get_cluster_by_sdp
% @export
% 
% Solver: SDPNAL+
function cluster_est = get_cluster_by_sdp(X, K)
D = X' * X;
Z_opt = kmeans_sdp_pengwei(D, K);
cluster_est = sdp_sol_to_cluster(Z_opt, K);
end
 
%% 
