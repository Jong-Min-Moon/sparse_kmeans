%% decide_stop_rand
% @export
% 
% |*Function Signature*|
% 
% is_stop = decide_stop_rand(rand_vec, loop_detect_start, window_size, min_delta)
% 
% |*Description*|
% 
% This function implements the early stopping logic for an iterative clustering 
% algorithm (like the one in fit_predict). It uses a history of Rand Index scores 
% (rand_vec) to determine if the process should halt due to either:
%% 
% # *Relative* Convergence *(stop_rand):* The Rand Index has stabilized, and 
% the recent change is below a minimum threshold.
% # *Loop Detection (stagnate_rand):* The algorithm is oscillating or stuck 
% in a short loop of previously visited states.
%% 
% If either condition is met, the function returns true and prints a summary 
% of the active stopping conditions.
% 
% |*Input Arguments*|
%% 
% *Argument*
%% 
% *Description*
%% 
% *Required Format*
%% 
% rand_vec
%% 
% A vector containing the historical Rand Index scores from previous iterations. 
% This is the primary input for assessing convergence.
%% 
% Numeric vector.
%% 
% loop_detect_start
%% 
% The minimum iteration number (index into rand_vec) from which loop detection 
% logic should begin running. This prevents spurious detection in the unstable 
% early iterations.
%% 
% Positive integer scalar.
%% 
% window_size
%% 
% The number of recent iterations to check when detecting stagnation or oscillation 
% (e.g., checking the last 5 scores).
%% 
% Positive integer scalar.
%% 
% min_delta
%% 
% The minimum required change in the Rand Index over the specified window. If 
% the change is less than this value, the algorithm is considered converged/stagnant.
%% 
% Non-negative numeric scalar (e.g., 1e-4).
%% 
% |*Output Argument*|
%% 
% *Output*
%% 
% *Description*
%% 
% *Format*
%% 
% is_stop
%% 
% A boolean flag indicating whether the iterative process should be terminated 
% early (true) or continue (false).
%% 
% Logical scalar (true or false).
%% 
% |*Internal Logic Flow*|
%% 
% # *Evaluate Convergence:* Calls a helper function (detect_relative_change 
% - _assumed to exist_) to check for stabilization of the Rand Index.
% # *Evaluate Oscillation:* Calls a helper function (detect_loop - _assumed 
% to exist_) to check for repeating patterns in the Rand Index scores.
% # *Stopping Decision:* If one or more stopping flags (stop_rand, stagnate_rand) 
% are set to true, is_stop is set to true, and the specific activated conditions 
% are printed to the console.
%% 
% 
function is_stop = decide_stop_rand(rand_vec, loop_detect_start, window_size, min_delta)
 is_stop = false;
        % Early stopping logic
        stop_rand = detect_relative_change(rand_vec, loop_detect_start, min_delta);
        stagnate_rand = detect_loop(rand_vec, loop_detect_start, window_size, min_delta);
        flags = [stop_rand, stagnate_rand];
        flag_names = {'stop_rand', 'stagnate_rand'};
        if sum(flags) >= 1
            fprintf('\nStopping early. Activated conditions:\n');
            for i = 1:length(flags)
                if flags(i)
                    fprintf('  • %s\n', flag_names{i});
                end
            end
            is_stop = true;
        end
 end
%% 
