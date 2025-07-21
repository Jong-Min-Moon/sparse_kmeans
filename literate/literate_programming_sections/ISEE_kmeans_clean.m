%% ISEE_kmeans_clean
% @export
% 
% 
% 
% % ISEE_kmeans_clean - Iterative clustering using ISEE-based refinement and 
% early stopping
% 
% %
% 
% % Inputs:
% 
% %   x                - Data matrix (p × n)
% 
% %   k                - Number of clusters
% 
% %   n_iter           - Maximum number of iterations
% 
% %   is_parallel      - Logical flag for parallel execution
% 
% %   loop_detect_start - Iteration to start loop detection
% 
% %   window_size      - Number of steps used for stagnation detection
% 
% %   min_delta        - Minimum improvement required to continue iterating
% 
% %
% 
% % Output:
% 
% %   cluster_estimate - Final cluster assignment (1 × n)
% 
% 
function cluster_estimate = ISEE_kmeans_clean(x, k, n_iter, is_parallel, loop_detect_start, window_size, min_delta)
    % Initialize tracking vectors
    obj_sdp = nan(1, n_iter);
    obj_lik = nan(1, n_iter);
    % Initial cluster assignment using spectral clustering
    cluster_estimate = cluster_spectral(x, k);
    for iter = 1:n_iter
        % One step of ISEE-based k-means refinement
        [cluster_estimate, s_hat,  obj_sdp(iter), obj_lik(iter)]  = ISEE_kmeans_clean_onestep(x, k, cluster_estimate, is_parallel);
        fprintf('Iteration %d | SDP obj: %.4f | Likelihood obj: %.4f\n', iter, obj_sdp(iter), obj_lik(iter));
        % Compute objective values
        
        % Early stopping condition
        is_stop = decide_stop(obj_sdp, obj_lik, loop_detect_start, window_size, min_delta);
        if is_stop
            break;
        end
    end
end
