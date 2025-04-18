function cluster_estimate = ISEE_kmeans_clean(x, k, n_iter, is_parallel, loop_detect_start, window_size, min_delta)
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
    % Initialize tracking vectors
    obj_sdp = nan(1, n_iter);
    obj_lik = nan(1, n_iter);
    % Initial cluster assignment using spectral clustering
    cluster_estimate = cluster_spectral(x, k);
    for iter = 1:n_iter
        % One step of ISEE-based k-means refinement
        [cluster_estimate, obj_sdp(iter), obj_lik(iter)]  = ISEE_kmeans_clean_onestep(x, k, cluster_estimate, is_parallel);
        fprintf('Iteration %d | SDP obj: %.4f | Likelihood obj: %.4f\n', iter, obj_sdp(iter), obj_lik(iter));
        % Compute objective values
        % Early stopping condition
        stop_sdp = detect_relative_change(obj_sdp, loop_detect_start, min_delta);
        stop_lik = detect_relative_change(obj_lik, loop_detect_start, min_delta);
        stagnate_sdp = detect_loop(obj_sdp, loop_detect_start, window_size, min_delta);
        stagnate_lik = detect_loop(obj_lik, loop_detect_start, window_size, min_delta);
% Collect all flags into a logical array
flags = [stop_sdp, stop_lik, stagnate_sdp, stagnate_lik];
flag_names = {'stop_sdp', 'stop_lik', 'stagnate_sdp', 'stagnate_lik'};
% Check if at least two conditions are true
if sum(flags) >= 2
    disp('\n Stopping early. Activated conditions:');
    for i = 1:length(flags)
        if flags(i)
            fprintf('  • %s\n', flag_names{i});
        end
    end
    break;
end
    end
end
